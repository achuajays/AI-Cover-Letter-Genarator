import streamlit as st
from groq import Groq
import base64
from io import BytesIO

# Attempt to import pdf2image for PDF conversion.
try:
    from pdf2image import convert_from_bytes
except ImportError:
    st.error("The pdf2image library is required to process PDF files. Install it with 'pip install pdf2image'.")


def prepare_resume_image(file_bytes, ext):
    """
    Converts the uploaded file bytes into a base64-encoded image string.
    For PDFs, it converts the first page to an image.
    For image files, it simply encodes the bytes.
    """
    if ext.endswith('.pdf'):
        try:
            images = convert_from_bytes(file_bytes)
            if images:
                # Convert the first page to JPEG format.
                image = images[0]
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            else:
                return None
        except Exception as e:
            st.error(f"PDF conversion error: {e}")
            return None
    else:
        return base64.b64encode(file_bytes).decode('utf-8')


def extract_text_from_resume_from_base64(base64_image):
    """
    Uses the GROQ vision model to extract text from the resume image provided as a base64 string.
    """
    client = Groq()
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all text from this resume image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }]
    try:
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.2-11b-vision-preview",  # Vision-enabled model.
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=False
        )
        extracted_text = response.choices[0].message.content
        return extracted_text
    except Exception as e:
        return f"Error extracting resume text: {e}"


def generate_cover_letter(resume_text, job_description, industry, tone):
    """
    Calls the GROQ text generation API to generate a tailored cover letter
    based on the extracted resume text and job details.
    """
    client = Groq()

    system_message = {
        "role": "system",
        "content": (
            "You are an expert cover letter generator. Using the candidate's resume details, "
            "job description, and industry-specific insights, generate a professional cover letter. "
            "Ensure the tone is aligned with the user's selection."
        )
    }

    user_message = {
        "role": "user",
        "content": (
            f"Generate a cover letter for a job in the {industry} industry. "
            f"Use the resume information below and the job description to highlight the candidate's strengths and fit for the role. "
            f"Ensure to incorporate relevant industry research and maintain a {tone} tone.\n\n"
            f"Resume:\n{resume_text}\n\n"
            f"Job Description:\n{job_description}\n"
        )
    }

    try:
        chat_completion = client.chat.completions.create(
            messages=[system_message, user_message],
            model="llama-3.3-70b-versatile",  # Text generation model.
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating cover letter: {e}"


def main():
    # --- Sidebar Settings for Theme & Template Selection ---
    st.sidebar.header("Settings")
    theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
    template_choice = st.sidebar.selectbox("Cover Letter Template", ["Classic", "Modern", "Creative"])

    # Apply a simple dark theme styling if chosen.
    if theme == "Dark":
        st.markdown(
            """
            <style>
            body { background-color: #333; color: white; }
            .css-18e3th9 { background-color: #333 !important; color: white !important; }
            </style>
            """,
            unsafe_allow_html=True
        )

    st.title("Cover Letter Generator from Resume")
    st.write(
        "Upload your resume (as an image or PDF) and provide the job details. "
        "The application will extract your resume text automatically and generate a tailored cover letter. "
        "You can choose different templates and themes for a personalized experience."
    )

    # --- Step 1: Resume Upload & Extraction ---
    st.subheader("Step 1: Upload Your Resume")
    uploaded_file = st.file_uploader("Upload your Resume (PNG, JPG, JPEG, or PDF)", type=["png", "jpg", "jpeg", "pdf"])

    extracted_resume_text = ""
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        ext = uploaded_file.name.lower()

        # Preview the uploaded file.
        if ext.endswith('.pdf'):
            try:
                images = convert_from_bytes(file_bytes)
                if images:
                    st.image(images[0], caption="First page of your PDF Resume", use_column_width=True)
                else:
                    st.error("Could not convert PDF to image.")
            except Exception as e:
                st.error(f"Error converting PDF: {e}")
        else:
            st.image(BytesIO(file_bytes), caption="Uploaded Resume", use_column_width=True)

        # Prepare a base64-encoded image.
        base64_image = prepare_resume_image(file_bytes, ext)
        if base64_image is None:
            st.error("Failed to process the uploaded file.")
        else:
            with st.spinner("Extracting resume text from the image..."):
                extracted_resume_text = extract_text_from_resume_from_base64(base64_image)
            if extracted_resume_text.startswith("Error"):
                st.error(extracted_resume_text)
            else:
                st.success("Resume text extracted successfully!")
                # Allow the user to edit the extracted text if needed.
                extracted_resume_text = st.text_area("Extracted Resume Text (Edit if needed):",
                                                     value=extracted_resume_text, height=200)
    else:
        st.info("Please upload your resume file to proceed.")

    # --- Step 2: Job Details ---
    st.subheader("Step 2: Enter Job Details")
    job_description = st.text_area("Job Description:", height=150)
    industry = st.text_input("Industry/Field:", "Technology")
    tone = st.selectbox("Select Tone:", ["Professional", "Friendly", "Formal", "Casual"])

    # --- Step 3: Generate Cover Letter ---
    if st.button("Generate Cover Letter"):
        if not extracted_resume_text.strip():
            st.error("Resume text is missing. Please check your upload and extraction.")
        elif not job_description.strip():
            st.error("Job description is required.")
        else:
            with st.spinner("Generating your cover letter..."):
                cover_letter = generate_cover_letter(extracted_resume_text, job_description, industry, tone)

            if cover_letter.startswith("Error"):
                st.error(cover_letter)
            else:
                # Display the cover letter in two tabs: Preview and Customize.
                tab1, tab2 = st.tabs(["Preview", "Customize"])
                with tab1:
                    st.markdown("### Generated Cover Letter")
                    if template_choice == "Modern":
                        st.markdown(f"<div style='font-family:Arial, sans-serif;'>{cover_letter}</div>",
                                    unsafe_allow_html=True)
                    elif template_choice == "Creative":
                        st.markdown(f"<div style='font-family:Georgia, serif; color:#4B0082;'>{cover_letter}</div>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(cover_letter)

                with tab2:
                    st.markdown("### Edit Your Cover Letter")
                    edited_cover_letter = st.text_area("Cover Letter Editor", cover_letter, height=300)
                    # Directly display the download button (no extra button click required)
                    st.download_button(
                        label="Download Edited Cover Letter",
                        data=edited_cover_letter,
                        file_name="cover_letter.txt",
                        mime="text/plain"
                    )


if __name__ == "__main__":
    main()
