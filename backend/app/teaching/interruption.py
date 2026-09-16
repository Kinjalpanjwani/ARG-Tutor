"""Interruption decisions live in TeachingEngine; this module documents the boundary.

Transport layers should cancel current audio using a session's response_revision,
then submit the transcript to TeachingEngine.respond(..., is_interruption=True).
"""
