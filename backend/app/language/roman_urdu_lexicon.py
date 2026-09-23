"""Keyword data for the cheap-first language detection pre-filter."""

ROMAN_URDU_LEXICON = frozenset({
    "aaj", "aap", "aapka", "aapke", "aapki", "aapko", "aapne", "aasaan", "aasan",
    "acha", "achha", "aaya", "aayi", "aik", "ab", "abhi", "aina", "aisa", "aisi", "alag",
    "alfaaz", "aya", "ayi", "bahar", "baat", "baatain", "baar", "bata", "bataiye",
    "batao", "behtar", "bhi", "bilkul", "bohat", "bohut", "bura", "chahiye", "chahta",
    "chahti", "chahte", "chhota", "chota", "dair", "dekh", "dekhein", "dekho", "dekhna",
    "dobara", "dhundho", "dushwar", "dushwaar", "ek", "faisla", "galat", "gaya", "gayi",
    "gaye", "ghalta", "hai", "hain", "hoga", "hogi", "hogay", "ho", "hoon", "hona",
    "hua", "hui", "hue", "insaaf", "itna", "itne", "jaana", "jaata", "jaati", "jaaye",
    "jaldi", "jana", "jis", "jo", "kab", "kabhi", "kafi", "kahin", "kahan", "kaheen",
    "kaise", "kaisi", "ka", "kal", "kam", "kar", "karna", "karne", "karta", "karti",
    "karte", "karo", "kaun", "kaunsi", "ke", "khaas", "kharab", "ki", "kisi", "kis",
    "ko", "kuch", "kuchh", "kya", "kyun", "kyon", "lekin", "liye", "magar", "maine",
    "mene", "mein", "mera", "meri", "mere", "mojood", "mujh", "mujhe", "mushkil",
    "nahi", "nahin", "nahee", "naya", "nayi", "paas", "pahada", "pahad", "pahra",
    "pahar", "phara", "parhna", "parho", "pata", "phir", "poora", "purana", "raat",
    "raha", "rahi", "rahe", "roz", "saath", "sath", "sab", "saare", "sahi", "sakta",
    "sakti", "sakte", "sakain", "samajh", "samajhna", "samjha", "samjhata",
    "samjhate", "sara", "se", "seekh", "seekhna", "shayad", "suno", "sunna", "tha",
    "thi", "thay", "theek", "teek", "thoda", "thodi", "thora", "toh", "tum",
    "tumhara", "tumhe", "tumko", "waqt", "wahan", "wahaan", "wala", "wali", "waly",
    "wapas", "wapis", "wo", "woh", "ya", "yahan", "yahaan", "yeh", "ye", "zaroor",
    "zyada",
})

ENGLISH_STOPWORDS = frozenset({
    "a", "about", "an", "and", "answer", "are", "as", "at", "be", "because", "book",
    "but", "by", "can", "chapter", "class", "could", "do", "does", "did", "example",
    "explain", "for", "from", "has", "have", "he", "help", "her", "hey", "hi", "his",
    "homework", "how", "i", "if", "in", "is", "it", "its", "learn", "lesson", "like",
    "me", "mean", "meaning", "means", "might", "my", "no", "not", "of", "ok",
    "okay", "on", "or", "our", "page", "please", "problem", "question", "school",
    "she", "should", "sir", "so", "solve", "study", "teach", "than", "thank", "thanks",
    "that", "the", "their", "them", "these", "they", "thing", "this", "those", "to",
    "understand", "us", "was", "we", "were", "what", "when", "where", "which", "who",
    "whom", "why", "will", "with", "would", "x", "yeah", "yes", "you", "your",
})

RU_CONFIDENT_RATIO = 0.5

EN_CONFIDENT_RATIO = 0.4