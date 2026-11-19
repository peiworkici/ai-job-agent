
from typing import TypedDict

class JobState(TypedDict):
    cv: str
    job_description: str

    cv_skills: list[str]
    job_skills: list[str]

    matched_skills: list[str]
    missing_skills: list[str]

    match_score: float
    recommendations: str
    cover_letter: str

    approved: bool

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

class SkillExtraction(BaseModel):
    cv_skills: list[str] = Field(
        description="Skills explicitly supported by the CV"
    )
    job_skills: list[str] = Field(
        description="Required skills from the job description"
    )

extractor = llm.with_structured_output(
    SkillExtraction
)



def extract_skills(state: JobState):

    result = extractor.invoke(
        f"""
        Extract technical and professional skills
        from the CV and Job Description.

        CV:
        {state['cv']}

        JOB DESCRIPTION:
        {state['job_description']}
        """
    )

    return {
        "cv_skills": result.cv_skills,
        "job_skills": result.job_skills
    }



def calculate_match(state: JobState):

    cv_skills = {
        skill.strip().lower()
        for skill in state["cv_skills"]
    }

    job_skills = {
        skill.strip().lower()
        for skill in state["job_skills"]
    }

    matched = cv_skills & job_skills
    missing = job_skills - cv_skills

    score = (
        len(matched) / len(job_skills) * 100
        if job_skills else 0
    )

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "match_score": round(score, 2)
    }


def generate_recommendations(state: JobState):

    prompt = f"""
    You are a professional career coach.

    Analyze the candidate's CV against the job
    description.

    Matched skills:
    {state['matched_skills']}

    Missing skills:
    {state['missing_skills']}

    Match score:
    {state['match_score']}%

    Based on the information provided:

    1. Suggest 3 CV improvement actions.
    2. Explain how to improve bullet points.
    3. Suggest keywords that could be added
       if supported by the candidate's experience.
    4. Never invent experience or qualifications.

    Be specific and practical.
    """

    response = llm.invoke(prompt)

    return {
        "recommendations": response.content
    }



def generate_cover_letter(state: JobState):

    prompt = f"""
    Write a professional English cover letter
    for this job application.

    CV:
    {state['cv']}

    Job Description:
    {state['job_description']}

    Matched skills:
    {state['matched_skills']}

    Improvement recommendations:
    {state['recommendations']}

    Requirements:
    - Professional tone
    - Maximum 300 words
    - Focus on relevant experience
    - Do not invent facts
    - Avoid generic AI-style expressions
    """

    response = llm.invoke(prompt)

    return {
        "cover_letter": response.content
    }


from langgraph.graph import StateGraph, START, END

def approval_router(state: JobState):
    if state["approved"]:
        return "draft"
    return END

builder = StateGraph(JobState)

builder.add_node("extract", extract_skills)
builder.add_node("match", calculate_match)
builder.add_node("recommend", generate_recommendations)
builder.add_node("draft", generate_cover_letter)

builder.add_edge(START, "extract")
builder.add_edge("extract", "match")
builder.add_edge("match", "recommend")

builder.add_conditional_edges(
    "recommend",
    approval_router,
    {
        "draft": "draft",
        END: END
    }
)

builder.add_edge("draft", END)

graph = builder.compile()

