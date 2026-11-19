import streamlit as st

from agent import graph


st.set_page_config(
    page_title="AI Job Application Agent",
    page_icon="💼"
)

st.title("💼 AI Job Application Agent")

st.caption(
    "Analyze your CV against a job description "
    "and generate tailored recommendations."
)

cv = st.text_area(
    "Paste your CV",
    height=250
)

jd = st.text_area(
    "Paste the Job Description",
    height=250
)

if st.button("Analyze My CV", type="primary"):

    if not cv.strip() or not jd.strip():
        st.warning("Please provide both CV and JD.")
    else:

        with st.spinner("Analyzing your profile..."):

            result = graph.invoke({
                "cv": cv,
                "job_description": jd,
                "approved": False
            })

            st.session_state["result"] = result

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    st.subheader("Match Score")

    st.metric(
        "Skill Coverage",
        f"{result['match_score']}%"
    )

    st.subheader("Matched Skills")

    st.write(result["matched_skills"])

    st.subheader("Missing Skills")

    st.write(result["missing_skills"])

    st.subheader("Recommendations")

    st.markdown(result["recommendations"])

    st.divider()

    st.subheader("Human Review")

    st.caption(
        "Review the recommendations before "
        "generating your cover letter."
    )

    if st.button("Approve & Generate Cover Letter"):

        with st.spinner("Generating cover letter..."):

            final_result = graph.invoke({
                "cv": cv,
                "job_description": jd,
                "cv_skills": result["cv_skills"],
                "job_skills": result["job_skills"],
                "matched_skills": result["matched_skills"],
                "missing_skills": result["missing_skills"],
                "match_score": result["match_score"],
                "recommendations": result["recommendations"],
                "approved": True
            })

            st.session_state["final_result"] = final_result

if "final_result" in st.session_state:

    st.subheader("Generated Cover Letter")

    st.markdown(
        st.session_state["final_result"]["cover_letter"]
    )
