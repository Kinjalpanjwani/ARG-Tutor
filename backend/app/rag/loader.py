from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image
from PyPDF2 import PdfReader


class DocumentLoadError(ValueError):
    pass


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    visual_paths: list[Path] = field(default_factory=list)


class DocumentLoader:
    """Loads text and retains page/slide visuals for multimodal course material."""

    def __init__(self, visuals_dir: Path | None = None) -> None:
        self.visuals_dir = visuals_dir

    def load(self, path: Path, document_id: str | None = None) -> list[PageText]:
        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                pages = self._pdf(path, document_id)
            elif suffix == ".pptx":
                pages = self._pptx(path, document_id)
            elif suffix in {".png", ".jpg", ".jpeg"}:
                pages = self._image(path, document_id)
            else:
                pages = [PageText(1, path.read_text(encoding="utf-8", errors="replace"))]
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(f"Could not read {path.name}: {exc}") from exc
        if not any(page.text.strip() for page in pages):
            raise DocumentLoadError("No readable text or visual content could be extracted")
        return pages

    def _visual_dir(self, document_id: str) -> Path:
        directory = (self.visuals_dir or Path("data/visuals")) / document_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _ocr(image: Path) -> str:
        if not shutil.which("tesseract"):
            raise DocumentLoadError("OCR is unavailable; install Tesseract to read image-based material")
        result = subprocess.run(
            ["tesseract", str(image), "stdout", "--psm", "6"],
            capture_output=True, text=True, timeout=90,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def _pdf(self, path: Path, document_id: str | None) -> list[PageText]:
        document_id = document_id or path.stem
        visual_dir = self._visual_dir(document_id)
        extracted = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
        process = subprocess.run(
            ["pdftoppm", "-png", "-r", "140", str(path), str(visual_dir / "page")],
            capture_output=True, text=True, timeout=180,
        )
        if process.returncode != 0:
            raise DocumentLoadError("PDF pages could not be rendered for visual processing")
        images = sorted(visual_dir.glob("page-*.png"))
        pages: list[PageText] = []
        for index, text in enumerate(extracted):
            image = images[index] if index < len(images) else None
            if len(text.strip()) < 30 and image:
                text = self._ocr(image)
            visual_note = " This page includes a retained source visual for diagrams, equations, tables, or figures."
            pages.append(PageText(index + 1, (text.strip() + visual_note).strip(), [image] if image else []))
        return pages

    def _image(self, path: Path, document_id: str | None) -> list[PageText]:
        document_id = document_id or path.stem
        visual = self._visual_dir(document_id) / f"image{path.suffix.lower()}"
        with Image.open(path) as image:
            image.convert("RGB").save(visual, quality=92)
        text = self._ocr(visual)
        return [PageText(1, f"{text}\nRetained source image containing visual course material.".strip(), [visual])]

    def _pptx(self, path: Path, document_id: str | None) -> list[PageText]:
        document_id = document_id or path.stem
        slides: dict[int, list[str]] = {}
        notes: dict[int, list[str]] = {}
        with ZipFile(path) as archive:
            for name in archive.namelist():
                match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
                note_match = re.fullmatch(r"ppt/notesSlides/notesSlide(\d+)\.xml", name)
                if match or note_match:
                    target = slides if match else notes
                    number = int((match or note_match).group(1))
                    xml = archive.read(name).decode("utf-8", errors="replace")
                    target[number] = [re.sub(r"\s+", " ", value).strip() for value in re.findall(r"<a:t>(.*?)</a:t>", xml) if value.strip()]
        visual_dir = self._visual_dir(document_id)
        with TemporaryDirectory() as temporary:
            conversion = subprocess.run(
                ["soffice", "--headless", "--convert-to", "pdf", "--outdir", temporary, str(path)],
                capture_output=True, text=True, timeout=180,
            )
            rendered: list[Path] = []
            converted = Path(temporary) / f"{path.stem}.pdf"
            if conversion.returncode == 0 and converted.exists():
                subprocess.run(["pdftoppm", "-png", "-r", "140", str(converted), str(visual_dir / "slide")], capture_output=True, timeout=180)
                rendered = sorted(visual_dir.glob("slide-*.png"))
        pages = []
        for number in sorted(slides):
            content = slides[number] + notes.get(number, [])
            image = rendered[number - 1] if number <= len(rendered) else None
            text = "\n".join(content).strip()
            if image and len(text) < 30:
                text = "\n".join(filter(None, [text, self._ocr(image)]))
            pages.append(PageText(number, f"Slide {number}. {text}\nRetained slide visual.".strip(), [image] if image else []))
        return pages
