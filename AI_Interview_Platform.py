"""
AI-Powered Interview Preparation Platform
ALL 9 PHASES IN ONE PYTHON FILE

Run:
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python main.py

Swagger: http://127.0.0.1:8000/docs

Phases included:
1. Architecture + embedded question dataset + SQLite database
2. FastAPI backend + catalog + database
3. Rule-based question selection
4. Interview session + answer/skip/complete workflow
5. Whisper/faster-whisper speech-to-text (lazy, optional; mock mode supported)
6. NLP analysis
7. Technical + behavioral evaluation
8. Computer vision webcam-frame analysis
9. Transparent scoring + feedback

AI limitations:
- Scores are practice heuristics, not scientifically validated hiring scores.
- Keyword/concept matching does not prove factual correctness.
- Text similarity is not guaranteed semantic or factual equivalence.
- Computer vision outputs observable indicators only; it does not infer emotion,
  personality, honesty, confidence, medical state, or psychological state.
- Webcam frames are processed in memory and are not stored by this application.
- Company questions in the dataset are generated practice questions, not claimed
  confidential/proprietary questions.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Depends, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker

# -----------------------------------------------------------------------------
# DATABASE
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "interview_platform.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

class Question(Base):
    __tablename__ = "questions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    company: Mapped[str] = mapped_column(String(100), nullable=False, default="General")
    interview_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    difficulty: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    skills: Mapped[str] = mapped_column(Text, nullable=False, default="")
    expected_concepts: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sample_answer_structure: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_type: Mapped[str] = mapped_column(String(80), nullable=False, default="Generated practice question")
    source_reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class InterviewSession(Base):
    __tablename__ = "interview_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(String(100), nullable=False)
    company: Mapped[str] = mapped_column(String(100), nullable=False)
    interview_type: Mapped[str] = mapped_column(String(50), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="active")
    current_question_index: Mapped[int] = mapped_column(Integer, default=0)
    total_questions: Mapped[int] = mapped_column(Integer, default=5)
    weak_skills: Mapped[str] = mapped_column(Text, default="[]")
    asked_question_ids: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

class InterviewQuestion(Base):
    __tablename__ = "interview_questions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("interview_sessions.id"), nullable=False)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"), nullable=False)
    question_number: Mapped[int] = mapped_column(Integer, nullable=False)
    asked_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    answered: Mapped[bool] = mapped_column(Boolean, default=False)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    __table_args__ = (UniqueConstraint("session_id", "question_number"),)

class Answer(Base):
    __tablename__ = "answers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    interview_question_id: Mapped[int] = mapped_column(ForeignKey("interview_questions.id"), nullable=False, unique=True)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_source: Mapped[str] = mapped_column(String(30), default="text")
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------------------------------------------------
# EMBEDDED 50-QUESTION PRACTICE DATA
# -----------------------------------------------------------------------------

RAW_QUESTIONS = [
("What is the difference between a list and a tuple in Python?","Python Developer","Technical Interview","Medium","Technical","Python","mutability, syntax, performance, use cases"),
("Explain Python dictionaries and their typical use cases.","Python Developer","Technical Interview","Easy","Technical","Python","key-value mapping, hashing, lookup"),
("What is the difference between == and is in Python?","Python Developer","Technical Interview","Easy","Technical","Python","value equality, object identity"),
("What is a Python generator?","Python Developer","Technical Interview","Medium","Technical","Python","yield, lazy evaluation, memory efficiency"),
("Explain exception handling in Python.","Python Developer","Technical Interview","Easy","Technical","Python","try, except, finally, raise"),
("What is normalization in a relational database?","Data Analyst","Technical Interview","Medium","Technical","SQL","redundancy, normal forms, integrity"),
("What is the difference between INNER JOIN and LEFT JOIN?","Data Analyst","Technical Interview","Easy","Technical","SQL","matching rows, unmatched rows"),
("What is an SQL index and what trade-offs does it have?","Data Analyst","Technical Interview","Medium","Technical","SQL","query performance, storage, write overhead"),
("What is overfitting and how can you reduce it?","Machine Learning Engineer","Technical Interview","Easy","Technical","Machine Learning","generalization, regularization, cross-validation"),
("Explain the bias-variance trade-off.","Machine Learning Engineer","Technical Interview","Hard","Technical","Machine Learning","bias, variance, generalization"),
("What is the difference between classification and regression?","Data Scientist","Technical Interview","Easy","Technical","Machine Learning","categorical target, continuous target"),
("Why do we split data into training and test sets?","Data Scientist","Technical Interview","Easy","Technical","Machine Learning","generalization, unseen data, evaluation"),
("What is cross-validation?","Data Scientist","Technical Interview","Medium","Technical","Machine Learning","folds, validation, model selection"),
("Explain precision, recall, and F1-score.","Data Scientist","Technical Interview","Medium","Technical","Machine Learning","precision, recall, F1-score"),
("What is gradient descent?","Machine Learning Engineer","Technical Interview","Medium","Technical","Machine Learning","loss, gradient, learning rate"),
("What is feature scaling?","Machine Learning Engineer","Technical Interview","Easy","Technical","Machine Learning","normalization, standardization"),
("Explain BFS and DFS.","Software Engineer","Technical Interview","Medium","Technical","Data Structures, Algorithms","queue, stack, graph traversal"),
("What is the time complexity of binary search?","Software Engineer","Technical Interview","Easy","Technical","Algorithms","sorted data, divide and conquer, O(log n)"),
("What is a hash table?","Software Engineer","Technical Interview","Medium","Technical","Data Structures","hash function, collision, lookup"),
("What is a REST API?","Backend Developer","Technical Interview","Easy","Technical","Web Development","HTTP, resources, statelessness, methods"),
("Tell me about yourself.","Software Engineer","HR Interview","Easy","HR","Communication","background, skills, career goal"),
("Why do you want this role?","Software Engineer","HR Interview","Easy","HR","Communication","motivation, skills, role alignment"),
("Why do you want to join our company?","Software Engineer","HR Interview","Medium","HR","Communication","company fit, role fit"),
("What are your strengths?","Software Engineer","HR Interview","Easy","HR","Communication","strength, evidence, relevance"),
("What is one weakness you are working on?","Software Engineer","HR Interview","Medium","HR","Communication","self-awareness, improvement plan"),
("Where do you see yourself in five years?","Software Engineer","HR Interview","Easy","HR","Communication","career direction, learning"),
("How do you handle criticism?","Software Engineer","HR Interview","Easy","HR","Communication","listening, reflection, action"),
("How do you manage multiple deadlines?","Software Engineer","HR Interview","Medium","HR","Communication","prioritization, planning"),
("Why should we hire you?","Software Engineer","HR Interview","Medium","HR","Communication","skills, evidence, fit"),
("Do you have any questions for us?","Software Engineer","HR Interview","Easy","HR","Communication","thoughtful questions, role interest"),
("Tell me about a time you solved a difficult problem.","Software Engineer","Behavioral Interview","Medium","Behavioral","Problem Solving","Situation, Task, Action, Result"),
("Describe a time you worked successfully in a team.","Software Engineer","Behavioral Interview","Easy","Behavioral","Leadership, Communication","Situation, team role, Action, Result"),
("Tell me about a time you made a mistake.","Software Engineer","Behavioral Interview","Medium","Behavioral","Communication","mistake, ownership, correction, learning"),
("Describe a conflict with a teammate and how you handled it.","Software Engineer","Behavioral Interview","Medium","Behavioral","Leadership, Communication","conflict, listening, resolution"),
("Tell me about a time you had to learn something quickly.","Software Engineer","Behavioral Interview","Easy","Behavioral","Communication","challenge, learning, application"),
("Describe a project where you took ownership.","Software Engineer","Behavioral Interview","Medium","Behavioral","Leadership","ownership, decisions, actions, result"),
("Tell me about a time you disagreed with a decision.","Software Engineer","Behavioral Interview","Hard","Behavioral","Leadership","disagreement, evidence, communication"),
("Describe a time you worked under pressure.","Software Engineer","Behavioral Interview","Medium","Behavioral","Communication","pressure, priorities, actions"),
("Tell me about a time your first solution did not work.","Software Engineer","Behavioral Interview","Medium","Behavioral","Problem Solving","failure, debugging, adaptation"),
("Give an example of receiving difficult feedback.","Software Engineer","Behavioral Interview","Medium","Behavioral","Communication","feedback, reflection, change"),
("How would you estimate the market size for online grocery delivery in India?","Business Analyst","Case/Business Interview","Hard","Case Study","Business Strategy","market definition, assumptions, top-down or bottom-up"),
("A product's sales dropped 20%. How would you investigate?","Business Analyst","Case/Business Interview","Medium","Case Study","Business Strategy","problem definition, segmentation, hypotheses"),
("How would you prioritize features for a new mobile app?","Product Manager","Case/Business Interview","Medium","Case Study","Business Strategy","user value, business value, effort"),
("What metrics would you track for a food-delivery app?","Product Manager","Case/Business Interview","Medium","Case Study","Business Strategy","acquisition, conversion, retention"),
("How would you launch a new product for college students?","Marketing Manager","Case/Business Interview","Medium","Case Study","Marketing","segmentation, positioning, channels"),
("What is customer lifetime value and why is it useful?","Marketing Manager","Case/Business Interview","Easy","Case Study","Marketing","revenue, retention, acquisition cost"),
("Explain the difference between revenue and profit.","Finance Analyst","Case/Business Interview","Easy","Case Study","Finance","revenue, cost, profit"),
("How would you evaluate whether a company should invest in a new project?","Finance Analyst","Case/Business Interview","Hard","Case Study","Finance","cash flows, NPV, risk, return"),
("A factory has rising costs. How would you identify the main drivers?","Operations Manager","Case/Business Interview","Medium","Case Study","Business Strategy","cost categories, process analysis, root cause"),
("How would you structure a consulting case about declining customer retention?","Consultant","Case/Business Interview","Hard","Case Study","Business Strategy","problem tree, hypotheses, segmentation, recommendation"),
]

def seed_questions(force: bool = False):
    with SessionLocal() as db:
        count = db.query(Question).count()
        if count and not force:
            return count
        if force:
            db.query(Answer).delete()
            db.query(InterviewQuestion).delete()
            db.query(InterviewSession).delete()
            db.query(Question).delete()
        for text, role, itype, diff, category, skills, concepts in RAW_QUESTIONS:
            db.add(Question(
                question_text=text, role=role, company="General", interview_type=itype,
                difficulty=diff, category=category, skills=skills, expected_concepts=concepts,
                sample_answer_structure="Answer clearly, explain the key idea, give an example, and connect it to the role.",
                source_type="Generated practice question"
            ))
        db.commit()
        return len(RAW_QUESTIONS)

seed_questions()

# -----------------------------------------------------------------------------
# COMMON HELPERS / SCHEMAS
# -----------------------------------------------------------------------------

ROLES = ["Software Engineer","Python Developer","Data Scientist","Data Analyst","Machine Learning Engineer","AI Engineer","Web/Backend/Frontend Developer","Business Analyst","Product Manager","Marketing Manager","Finance Analyst","HR Manager","Operations Manager","Consultant"]
COMPANIES = ["General","Google","Microsoft","Amazon","Deloitte","Accenture","TCS","Infosys","Wipro","IBM"]
INTERVIEW_TYPES = ["Technical Interview","HR Interview","Behavioral Interview","Case/Business Interview","Managerial Interview","Mixed Interview"]
DIFFICULTIES = ["Easy","Medium","Hard"]

class StartInterviewRequest(BaseModel):
    role: str
    company: str = "General"
    interview_type: str
    difficulty: str = "Medium"
    total_questions: int = Field(default=5, ge=1, le=20)
    weak_skills: list[str] = Field(default_factory=list)

class AnswerRequest(BaseModel):
    answer_text: str = Field(min_length=1)
    answer_source: str = "text"

class NLPRequest(BaseModel):
    answer: str
    expected_concepts: str = ""
    reference_answer: str = ""

class TechnicalRequest(BaseModel):
    answer: str
    expected_concepts: str = ""
    reference_answer: str = ""
    skills: str = ""

class BehavioralRequest(BaseModel):
    answer: str
    expected_concepts: str = ""
    question: str = ""

class VisualMetrics(BaseModel):
    face_detected: bool = False
    face_count: int = 0
    face_centered: bool = False
    eye_visibility: str = "unknown"
    head_orientation: str = "unknown"
    attention_indicator: float = 0.0

class ScoreRequest(BaseModel):
    interview_type: str = "Technical Interview"
    technical: float = Field(default=0, ge=0, le=100)
    relevance: float = Field(default=0, ge=0, le=100)
    communication: float = Field(default=0, ge=0, le=100)
    structure: float = Field(default=0, ge=0, le=100)
    visual: float = Field(default=0, ge=0, le=100)
    technical_weight: float = Field(default=0.40, ge=0, le=1)
    relevance_weight: float = Field(default=0.20, ge=0, le=1)
    communication_weight: float = Field(default=0.20, ge=0, le=1)
    structure_weight: float = Field(default=0.10, ge=0, le=1)
    visual_weight: float = Field(default=0.10, ge=0, le=1)

# -----------------------------------------------------------------------------
# PHASE 3: RULE-BASED QUESTION SELECTION
# -----------------------------------------------------------------------------

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def split_values(s: str) -> list[str]:
    return [x.strip() for x in re.split(r",|;", s or "") if x.strip()]

def question_score(q: Question, role: str, interview_type: str, difficulty: str, company: str, weak_skills: list[str], asked: set[int]):
    if q.id in asked:
        return -999, ["already asked"]
    score = 0
    reasons = []
    if norm(q.role) == norm(role): score += 40; reasons.append("role match")
    if norm(q.interview_type) == norm(interview_type): score += 25; reasons.append("interview type match")
    if norm(q.difficulty) == norm(difficulty): score += 15; reasons.append("difficulty match")
    if norm(q.company) == norm(company): score += 5; reasons.append("company match")
    if norm(q.company) == "general": score += 2; reasons.append("general practice question")
    weak = {norm(x) for x in weak_skills}
    for skill in split_values(q.skills):
        if norm(skill) in weak:
            score += 5
    if weak and any(norm(s) in weak for s in split_values(q.skills)):
        reasons.append("targets a weak skill")
    return score, reasons

def select_question(db: Session, session: InterviewSession):
    questions = list(db.scalars(select(Question)).all())
    asked = set(json.loads(session.asked_question_ids or "[]"))
    weak = json.loads(session.weak_skills or "[]")
    ranked = [(question_score(q, session.role, session.interview_type, session.difficulty, session.company, weak, asked), q) for q in questions]
    ranked = [(s, q) for s, q in ranked if s[0] >= 0]
    if not ranked:
        return None, None
    ranked.sort(key=lambda item: (-item[0][0], item[1].id))
    return ranked[0][1], ranked[0][0]

# -----------------------------------------------------------------------------
# PHASE 6: LIGHTWEIGHT NLP
# -----------------------------------------------------------------------------

STOPWORDS = {"the","a","an","is","are","was","were","to","of","and","or","in","on","for","with","this","that","it","as","by","be","can","do","does","how","what","why","i","we","you","they","he","she"}
FILLERS = {"um","uh","like","actually","basically","you know","sort of","kind of"}

def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9+#.]+", text.lower())

def concept_terms(concepts: str) -> list[str]:
    return [norm(x) for x in split_values(concepts)]

def concept_found(answer: str, concept: str) -> bool:
    a = norm(answer)
    c = norm(concept)
    aliases = {
        "mutability": ["mutable", "immutable", "mutability"],
        "syntax": ["square brackets", "parentheses", "syntax"],
        "performance": ["performance", "memory", "speed"],
        "use cases": ["use case", "use cases", "when to use"],
        "value equality": ["==", "value equality", "equal"],
        "object identity": [" is ", "identity", "same object"],
        "normal forms": ["1nf", "2nf", "3nf", "normal form"],
        "generalization": ["generalization", "unseen data", "test data"],
        "regularization": ["regularization", "l1", "l2", "dropout"],
        "cross-validation": ["cross validation", "cross-validation", "fold"],
        "precision": ["precision"], "recall": ["recall"], "f1-score": ["f1", "f1-score"],
        "situation": ["situation", "context", "when"], "task": ["task", "responsibility", "goal"],
        "action": ["i did", "i took", "action", "implemented", "organized"],
        "result": ["result", "outcome", "improved", "reduced", "increased"],
    }
    options = aliases.get(c, [c])
    return any(o in a for o in options)

def cosine_text(a: str, b: str) -> float:
    ta = [x for x in tokens(a) if x not in STOPWORDS]
    tb = [x for x in tokens(b) if x not in STOPWORDS]
    if not ta or not tb: return 0.0
    ca, cb = {}, {}
    for x in ta: ca[x] = ca.get(x, 0) + 1
    for x in tb: cb[x] = cb.get(x, 0) + 1
    keys = set(ca) | set(cb)
    dot = sum(ca.get(k,0)*cb.get(k,0) for k in keys)
    da = math.sqrt(sum(v*v for v in ca.values()))
    db = math.sqrt(sum(v*v for v in cb.values()))
    return round(dot/(da*db), 4) if da and db else 0.0

def nlp_analyze(answer: str, expected: str = "", reference: str = "") -> dict[str, Any]:
    answer = answer.strip()
    words = tokens(answer)
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
    concepts = concept_terms(expected)
    covered = [c for c in concepts if concept_found(answer, c)]
    missing = [c for c in concepts if c not in covered]
    coverage = round(100 * len(covered) / len(concepts), 2) if concepts else 0.0
    relevance = round(100 * cosine_text(answer, reference), 2) if reference else min(100.0, coverage)
    completeness = round(min(100.0, 0.65 * coverage + 35.0 * (1 if len(words) >= 15 else len(words)/15)), 2)
    conciseness = round(max(0.0, 100 - max(0, len(words)-80)*0.7), 2)
    avg_sentence = len(words) / max(1, len(sentences))
    clarity = round(max(0.0, min(100.0, 100 - max(0, avg_sentence-25)*2 - (8 if len(sentences)==0 else 0))), 2)
    unique = len(set(words))
    diversity = round(unique / len(words), 3) if words else 0.0
    filler_counts = {f: len(re.findall(r"\b"+re.escape(f)+r"\b", answer.lower())) for f in FILLERS}
    filler_counts = {k:v for k,v in filler_counts.items() if v}
    return {
        "relevance": relevance,
        "semantic_similarity": round(cosine_text(answer, reference), 4) if reference else 0.0,
        "technical_concept_coverage": coverage,
        "covered_concepts": covered,
        "missing_concepts": missing,
        "completeness": completeness,
        "conciseness": conciseness,
        "clarity": clarity,
        "structure": {"sentence_count": len(sentences), "word_count": len(words), "avg_words_per_sentence": round(avg_sentence,2), "vocabulary_diversity": diversity},
        "filler_words": filler_counts,
        "limitations": ["These are transparent practice estimates, not scientifically validated scores.", "Concept matching cannot prove factual correctness.", "TF-style cosine similarity measures word overlap and is not guaranteed semantic or factual equivalence."],
    }

# -----------------------------------------------------------------------------
# PHASE 7: TECHNICAL + BEHAVIORAL EVALUATION
# -----------------------------------------------------------------------------

def technical_evaluate(req: TechnicalRequest):
    nlp = nlp_analyze(req.answer, req.expected_concepts, req.reference_answer)
    coverage = nlp["technical_concept_coverage"]
    similarity = nlp["semantic_similarity"] * 100
    completeness = nlp["completeness"]
    score = round(0.50*coverage + 0.30*similarity + 0.20*completeness, 2) if req.reference_answer else round(0.70*coverage + 0.30*completeness, 2)
    return {
        "evaluation_type": "technical",
        "technical_score": score,
        "concept_coverage": coverage,
        "covered_concepts": nlp["covered_concepts"],
        "missing_concepts": nlp["missing_concepts"],
        "reference_overlap": similarity,
        "skills": split_values(req.skills),
        "strengths": ["Good coverage of expected concepts."] if coverage >= 70 else [],
        "improvements": ["Review the missing concepts and explain them with an example."] if nlp["missing_concepts"] else ["Add a concrete example or trade-off to strengthen the answer."],
        "limitations": nlp["limitations"],
    }

def behavioral_evaluate(req: BehavioralRequest):
    a = norm(req.answer)
    situation = any(x in a for x in ["when ", "during ", "in my project", "the situation", "at college", "we had"])
    task = any(x in a for x in ["my task", "my responsibility", "i had to", "the goal was", "i was responsible"])
    action = any(x in a for x in ["i did", "i took", "i implemented", "i organized", "i decided", "i communicated", "i created"])
    result = any(x in a for x in ["result", "outcome", "improved", "reduced", "increased", "achieved", "%", "learned"])
    star = {"situation": situation, "task": task, "action": action, "result": result}
    star_score = sum(star.values()) * 25
    specificity = min(100, 35 + min(65, len(tokens(req.answer))*1.8))
    relevance = 100 if not req.question else min(100, 60 + 40*(1 if any(x in a for x in tokens(req.question)[:8] if len(x)>4) else 0))
    score = round(0.45*star_score + 0.30*specificity + 0.25*relevance, 2)
    improvements = []
    if not situation: improvements.append("Start with the context or Situation.")
    if not task: improvements.append("State your specific Task or responsibility.")
    if not action: improvements.append("Explain what you personally did, not only what the team did.")
    if not result: improvements.append("Finish with a measurable Result or learning.")
    return {"evaluation_type":"behavioral","behavioral_score":score,"star_score":star_score,"star":star,"specificity":round(specificity,2),"relevance":round(relevance,2),"improvements":improvements or ["Good STAR coverage. Add measurable outcomes where possible."],"limitations":["STAR detection is based on text cues and cannot fully understand the quality of a real-life example."]}

# -----------------------------------------------------------------------------
# PHASE 8: OPENCV COMPUTER VISION
# -----------------------------------------------------------------------------

def analyze_image_bytes(image_bytes: bytes) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise HTTPException(500, "opencv-python and numpy are required for visual analysis")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode the uploaded image")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    h, w = gray.shape[:2]
    face_detected = len(faces) > 0
    centered = False
    eye_visibility = "no_face"
    orientation = "unknown"
    if face_detected:
        x,y,fw,fh = max(faces, key=lambda f: f[2]*f[3])
        cx, cy = x+fw/2, y+fh/2
        centered = abs(cx-w/2)/(w/2) < 0.22 and abs(cy-h/2)/(h/2) < 0.30
        roi = gray[y:y+fh, x:x+fw]
        eyes = eye_cascade.detectMultiScale(roi, 1.1, 5, minSize=(15,15))
        eye_visibility = "two_eyes_detected" if len(eyes)>=2 else ("one_eye_detected" if len(eyes)==1 else "eyes_not_detected")
        orientation = "approximately_forward" if centered else ("approximately_side_or_off_center" if abs(cx-w/2)/(w/2)>0.35 else "approximately_forward")
    attention = 0.0
    if face_detected:
        attention = 40.0
        if centered: attention += 35
        if eye_visibility == "two_eyes_detected": attention += 25
        elif eye_visibility == "one_eye_detected": attention += 12
    return {"face_detected":face_detected,"face_count":len(faces),"face_centered":centered,"eye_visibility":eye_visibility,"head_orientation":orientation,"attention_indicator":round(attention,2),"image_width":w,"image_height":h,"limitations":["This is an observable computer-vision proxy, not emotion or psychological-state detection.","Lighting, glasses, camera angle, image quality, and face position can affect detection.","The uploaded frame is processed in memory and not stored by this application."]}

# -----------------------------------------------------------------------------
# PHASE 9: SCORING + FEEDBACK
# -----------------------------------------------------------------------------

def final_score(req: ScoreRequest):
    weights = {"technical":req.technical_weight,"relevance":req.relevance_weight,"communication":req.communication_weight,"structure":req.structure_weight,"visual":req.visual_weight}
    total_w = sum(weights.values())
    if abs(total_w-1.0) > 0.001:
        raise HTTPException(400, f"Weights must total 1.0 (100%). Current total: {total_w:.3f}")
    values = {"technical":req.technical,"relevance":req.relevance,"communication":req.communication,"structure":req.structure,"visual":req.visual}
    weighted = {k: round(values[k]*weights[k],2) for k in values}
    overall = round(sum(weighted.values()),2)
    strengths = [f"{k.title()} performance is strong ({values[k]:.1f}/100)." for k in values if values[k]>=75]
    improvements = [f"Practice {k} because the current score is {values[k]:.1f}/100." for k in values if values[k]<60]
    recommendations = []
    if req.technical < 60: recommendations.append("Review core technical concepts and practice explaining them with examples.")
    if req.communication < 60: recommendations.append("Practice concise answers, clear sentences, and reduce unnecessary filler words.")
    if req.structure < 60: recommendations.append("Use a clear beginning, key points, example, and conclusion; use STAR for behavioral questions.")
    if req.visual < 60: recommendations.append("Check camera framing, lighting, and approximate eye contact during practice.")
    if not recommendations: recommendations.append("Maintain current strengths and practice harder questions to improve consistency.")
    return {"overall_score":overall,"component_scores":values,"weights":weights,"weighted_contributions":weighted,"strengths":strengths,"areas_for_improvement":improvements,"recommendations":recommendations,"limitations":["The weights are configurable engineering heuristics and are not scientifically validated.","This score is for interview practice and is not a hiring decision or prediction."]}

# -----------------------------------------------------------------------------
# FASTAPI APP / ROUTES
# -----------------------------------------------------------------------------

app = FastAPI(title="AI-Powered Interview Preparation Platform", version="1.0.0", description="All 9 project phases consolidated into one Python file.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"message":"AI Interview Platform is running","phases":"1-9","docs":"/docs"}

@app.get("/api/health")
def health():
    return {"status":"ok","phases":"1-9","database":str(DB_PATH)}

@app.get("/api/catalog/roles")
def roles(): return {"roles":ROLES}
@app.get("/api/catalog/companies")
def companies(): return {"companies":COMPANIES}
@app.get("/api/catalog/interview-types")
def interview_types(): return {"interview_types":INTERVIEW_TYPES}
@app.get("/api/catalog/difficulties")
def difficulties(): return {"difficulties":DIFFICULTIES}

@app.get("/api/questions")
def list_questions(role: Optional[str]=None, company: Optional[str]=None, interview_type: Optional[str]=None, difficulty: Optional[str]=None, category: Optional[str]=None, skill: Optional[str]=None, limit: int=20, db: Session=Depends(get_db)):
    qs = list(db.scalars(select(Question)).all())
    if role: qs=[q for q in qs if norm(q.role)==norm(role)]
    if company and norm(company)!="general": qs=[q for q in qs if norm(q.company)==norm(company) or norm(q.company)=="general"]
    if interview_type: qs=[q for q in qs if norm(q.interview_type)==norm(interview_type)]
    if difficulty: qs=[q for q in qs if norm(q.difficulty)==norm(difficulty)]
    if category: qs=[q for q in qs if norm(q.category)==norm(category)]
    if skill: qs=[q for q in qs if norm(skill) in [norm(x) for x in split_values(q.skills)]]
    return {"count":len(qs),"questions":[question_dict(q) for q in qs[:max(1,min(limit,100))]]}

def question_dict(q: Question):
    return {"id":q.id,"question_text":q.question_text,"role":q.role,"company":q.company,"interview_type":q.interview_type,"difficulty":q.difficulty,"category":q.category,"skills":q.skills,"expected_concepts":q.expected_concepts,"sample_answer_structure":q.sample_answer_structure,"source_type":q.source_type,"source_reference":q.source_reference}

def current_question(db: Session, session: InterviewSession):
    iq = db.scalar(select(InterviewQuestion).where(InterviewQuestion.session_id==session.id, InterviewQuestion.question_number==session.current_question_index+1))
    if iq: return iq
    q, meta = select_question(db, session)
    if not q: return None
    iq=InterviewQuestion(session_id=session.id, question_id=q.id, question_number=session.current_question_index+1)
    db.add(iq)
    asked=json.loads(session.asked_question_ids or "[]")
    if q.id not in asked: asked.append(q.id)
    session.asked_question_ids=json.dumps(asked)
    db.commit(); db.refresh(iq)
    return iq

def question_payload(db, iq):
    q=db.get(Question,iq.question_id)
    return {"interview_question_id":iq.id,"question_number":iq.question_number,"question":q.question_text,"role":q.role,"company":q.company,"category":q.category,"skills":q.skills,"expected_concepts":q.expected_concepts,"difficulty":q.difficulty,"source_type":q.source_type}

@app.post("/api/interview/start")
def start_interview(req: StartInterviewRequest, db: Session=Depends(get_db)):
    s=InterviewSession(role=req.role,company=req.company,interview_type=req.interview_type,difficulty=req.difficulty,total_questions=req.total_questions,weak_skills=json.dumps(req.weak_skills))
    db.add(s); db.commit(); db.refresh(s)
    iq=current_question(db,s)
    if not iq:
        db.delete(s); db.commit(); raise HTTPException(404,"No suitable practice question is available.")
    return {"session_id":s.id,"status":s.status,"total_questions":s.total_questions,"current":question_payload(db,iq)}

@app.get("/api/interview/{session_id}")
def get_interview(session_id:int, db: Session=Depends(get_db)):
    s=db.get(InterviewSession,session_id)
    if not s: raise HTTPException(404,"Interview session not found")
    answers=db.scalars(select(Answer).join(InterviewQuestion).where(InterviewQuestion.session_id==s.id)).all()
    return {"session_id":s.id,"role":s.role,"company":s.company,"interview_type":s.interview_type,"difficulty":s.difficulty,"status":s.status,"current_question_number":s.current_question_index+1,"total_questions":s.total_questions,"answered_questions":len(answers),"asked_question_ids":json.loads(s.asked_question_ids or "[]")}

@app.get("/api/interview/{session_id}/current")
def get_current_question(session_id:int, db: Session=Depends(get_db)):
    s=db.get(InterviewSession,session_id)
    if not s: raise HTTPException(404,"Interview session not found")
    if s.status!="active": raise HTTPException(400,"Interview is not active")
    iq=current_question(db,s)
    if not iq: raise HTTPException(404,"No question available")
    return question_payload(db,iq)

@app.post("/api/interview/{session_id}/answer")
def submit_answer(session_id:int, req:AnswerRequest, db: Session=Depends(get_db)):
    s=db.get(InterviewSession,session_id)
    if not s: raise HTTPException(404,"Interview session not found")
    if s.status!="active": raise HTTPException(400,"Interview is not active")
    iq=current_question(db,s)
    if not iq: raise HTTPException(400,"No current question")
    if db.scalar(select(Answer).where(Answer.interview_question_id==iq.id)): raise HTTPException(409,"Current question has already been answered")
    db.add(Answer(interview_question_id=iq.id,answer_text=req.answer_text.strip(),answer_source=req.answer_source))
    iq.answered=True; s.current_question_index+=1
    completed=s.current_question_index>=s.total_questions
    if completed: s.status="completed"; s.completed_at=datetime.utcnow()
    db.commit()
    next_q=None if completed else current_question(db,s)
    return {"saved":True,"completed":completed,"next":question_payload(db,next_q) if next_q else None}

@app.post("/api/interview/{session_id}/skip")
def skip_question(session_id:int, db: Session=Depends(get_db)):
    s=db.get(InterviewSession,session_id)
    if not s: raise HTTPException(404,"Interview session not found")
    if s.status!="active": raise HTTPException(400,"Interview is not active")
    iq=current_question(db,s)
    if not iq: raise HTTPException(400,"No current question")
    iq.skipped=True; s.current_question_index+=1
    completed=s.current_question_index>=s.total_questions
    if completed: s.status="completed"; s.completed_at=datetime.utcnow()
    db.commit()
    next_q=None if completed else current_question(db,s)
    return {"skipped":True,"completed":completed,"next":question_payload(db,next_q) if next_q else None}

@app.post("/api/interview/{session_id}/complete")
def complete_interview(session_id:int, db: Session=Depends(get_db)):
    s=db.get(InterviewSession,session_id)
    if not s: raise HTTPException(404,"Interview session not found")
    s.status="completed"; s.completed_at=datetime.utcnow(); db.commit()
    return {"session_id":s.id,"status":s.status}

@app.post("/api/analyze/answer")
def analyze_answer(req:NLPRequest): return nlp_analyze(req.answer,req.expected_concepts,req.reference_answer)

@app.post("/api/evaluate/technical")
def evaluate_technical(req:TechnicalRequest): return technical_evaluate(req)

@app.post("/api/evaluate/behavioral")
def evaluate_behavioral(req:BehavioralRequest): return behavioral_evaluate(req)

@app.post("/api/analyze/visual")
async def analyze_visual(file: UploadFile=File(...)):
    image=await file.read()
    if not image: raise HTTPException(400,"Uploaded image is empty")
    return analyze_image_bytes(image)

@app.post("/api/score/interview")
def score_interview(req:ScoreRequest): return final_score(req)

# -----------------------------------------------------------------------------
# PHASE 5: SPEECH-TO-TEXT
# -----------------------------------------------------------------------------

ALLOWED_AUDIO={".wav",".mp3",".m4a",".mp4",".webm",".ogg",".flac"}

@app.post("/api/speech/transcribe")
async def transcribe_audio(file: UploadFile=File(...)):
    filename=file.filename or "audio.wav"
    ext="."+filename.rsplit(".",1)[-1].lower() if "." in filename else ".wav"
    if ext not in ALLOWED_AUDIO: raise HTTPException(400,f"Unsupported audio format. Allowed: {', '.join(sorted(ALLOWED_AUDIO))}")
    data=await file.read()
    if not data: raise HTTPException(400,"Uploaded audio file is empty")
    if len(data)>25*1024*1024: raise HTTPException(413,"Audio file is larger than 25 MB")
    if os.getenv("WHISPER_MOCK","false").lower()=="true":
        return {"transcript":"[MOCK TRANSCRIPT] Replace with a real recorded answer.","language":"mock","duration_seconds":None,"model":"mock"}
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(500,"faster-whisper is not installed. Install it with: pip install faster-whisper")
    model_size=os.getenv("WHISPER_MODEL","tiny")
    device=os.getenv("WHISPER_DEVICE","cpu")
    compute=os.getenv("WHISPER_COMPUTE_TYPE","int8")
    path=None
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix=ext) as tmp:
            tmp.write(data); path=tmp.name
        model=WhisperModel(model_size,device=device,compute_type=compute)
        segments,info=model.transcribe(path,language=(os.getenv("WHISPER_LANGUAGE") or None),vad_filter=True)
        segments=list(segments)
        transcript=" ".join(s.text.strip() for s in segments).strip()
        duration=getattr(info,"duration",None)
        return {"transcript":transcript,"language":getattr(info,"language",None),"duration_seconds":round(duration,2) if duration else None,"model":model_size}
    except Exception as exc:
        raise HTTPException(500,f"Transcription failed: {exc}")
    finally:
        if path:
            try: Path(path).unlink(missing_ok=True)
            except OSError: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
