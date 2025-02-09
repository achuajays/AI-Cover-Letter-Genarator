import streamlit as st
from groq import Groq


def generate_cover_letter(resume_text, job_description, industry, tone):
    """
    Calls the GROQ chat completion API to generate a tailored cover letter.
    """
    client = Groq()

    # Construct messages for the conversation.
    system_message = {
        "role": "system",
        "content": (
            "You are an expert cover letter generator that tailors letters based on the candidate's resume, "
            "the job description, and incorporates research about the industry. "
            "Ensure the output is professional and aligned with the chosen tone."
        )
    }

    # Create a detailed prompt that includes all the necessary context.
    user_message = {
        "role": "user",
        "content": (
            f"Generate a cover letter for a job in the {industry} industry. "
            f"Use the resume information below and the job description to highlight the candidate's strengths and fit for the role. "
            f"Make sure to incorporate any relevant industry research and align the tone with a {tone} style.\n\n"
            f"Resume:\n{resume_text}\n\n"
            f"Job Description:\n{job_description}\n"
        )
    }

    try:
        # Call the GROQ chat completion API.
        chat_completion = client.chat.completions.create(
            messages=[system_message, user_message],
            model="llama-3.3-70b-versatile",  # Use an appropriate model for text generation.
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )
        # Return the generated cover letter text.
        return chat_completion.choices[0].message.content
    except Exception as e:
        # In production, consider logging the error.
        return f"An error occurred while generating the cover letter: {e}"


def main():
    st.title("Cover Letter Generator from Resume")
    st.write(
        "Generate a professional, tailored cover letter based on your resume and the job description. "
        "Fill in the fields below and click the button to generate your cover letter."
    )

    # Input areas for resume, job description, industry, and tone.
    resume_text = st.text_area("Paste your Resume text here:", height=200)
    job_description = st.text_area("Paste the Job Description here:", height=150)
    industry = st.text_input("Industry/Field", "Technology")
    tone = st.selectbox("Select Tone", ["Professional", "Friendly", "Formal", "Casual"])

    # Button to trigger cover letter generation.
    if st.button("Generate Cover Letter"):
        if not resume_text.strip() or not job_description.strip():
            st.error("Please provide both your resume and the job description.")
        else:
            with st.spinner("Generating your cover letter..."):
                cover_letter = generate_cover_letter(resume_text, job_description, industry, tone)

            if cover_letter.startswith("An error occurred"):
                st.error(cover_letter)
            else:
                st.success("Cover letter generated successfully!")
                st.text_area("Your Generated Cover Letter", cover_letter, height=300)
                st.download_button(
                    label="Download Cover Letter",
                    data=cover_letter,
                    file_name="cover_letter.txt",
                    mime="text/plain",
                )


if __name__ == "__main__":
    main()
