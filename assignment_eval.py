import os
import io
import json
import boto3
import streamlit as st
import pandas as pd
import PyPDF2
import docx

class FileParser:
    @staticmethod
    def parse_pdf(uploaded_file):
        """
        Parse content from PDF file
        
        Args:
            uploaded_file: Streamlit uploaded file object
        
        Returns:
            str: Extracted text from PDF
        """
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error parsing PDF: {e}")
            return ""

    @staticmethod
    def parse_docx(uploaded_file):
        """
        Parse content from DOCX file
        
        Args:
            uploaded_file: Streamlit uploaded file object
        
        Returns:
            str: Extracted text from DOCX
        """
        try:
            doc = docx.Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            st.error(f"Error parsing DOCX: {e}")
            return ""

class BedrockEvaluator:
    def __init__(self, 
                 region_name='us-east-1', 
                 model_id='anthropic/claude-v2:1'):
        """
        Initialize Bedrock client for assignment evaluation
        
        Args:
            region_name (str): AWS region for Bedrock
            model_id (str): Bedrock model identifier
        """
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime', 
            region_name=region_name
        )
        self.model_id = model_id

    def generate_comprehensive_prompt(self, 
                                      assignment_content: str, 
                                      criteria_content: str) -> str:
        """
        Generate a detailed evaluation prompt
        
        Args:
            assignment_content (str): Content of the submitted assignment
            criteria_content (str): Evaluation criteria description
        
        Returns:
            str: Comprehensive evaluation prompt
        """
        prompt = f"""You are an expert academic evaluator tasked with providing a comprehensive assessment of an assignment.

EVALUATION CRITERIA:
{criteria_content}

ASSIGNMENT CONTENT:
{assignment_content}

EVALUATION INSTRUCTIONS:
1. Carefully analyze the assignment against each criterion
2. Provide a detailed score for each criterion
3. Give specific, constructive feedback
4. Suggest concrete improvements
5. Calculate a total weighted score

Please structure your response as a detailed JSON with the following format:
{{
    "criterion_breakdown": [
        {{
            "name": "Criterion Name",
            "max_points": 20,
            "awarded_points": 15,
            "specific_feedback": "Detailed comments on performance",
            "improvement_suggestions": "Specific recommendations"
        }}
    ],
    "total_score": 85,
    "overall_grade": "B+",
    "overall_feedback": "Comprehensive summary of assignment quality",
    "key_strengths": ["Strength 1", "Strength 2"],
    "areas_for_improvement": ["Improvement Area 1", "Improvement Area 2"]
}}

Ensure the response is comprehensive, objective, and provides actionable insights."""
        return prompt

    def invoke_bedrock_model(self, prompt: str) -> dict:
        """
        Invoke Bedrock model for assignment evaluation
        
        Args:
            prompt (str): Evaluation prompt
        
        Returns:
            dict: Parsed evaluation results
        """
        try:
            # Anthropic Claude-specific body structure
            body = json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 4096,
                "temperature": 0.3,
                "top_p": 0.9,
                "stop_sequences": ["\n\nHuman:"]
            })

            # Invoke Bedrock model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body
            )

            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract and parse completion
            completion = response_body.get('completion', '{}')
            
            # Attempt to parse JSON, with fallback
            try:
                evaluation = json.loads(completion)
            except json.JSONDecodeError:
                # Fallback parsing if JSON decode fails
                evaluation = {
                    "overall_feedback": completion,
                    "total_score": "Unable to parse exact score"
                }
            
            return evaluation

        except Exception as e:
            st.error(f"Bedrock Model Invocation Error: {e}")
            return {}

def main():
    st.set_page_config(
        page_title="🎓 AI-Powered Assignment Evaluator", 
        page_icon="📝",
        layout="wide"
    )

    # Title and Description
    st.title("📝 AI-Powered Assignment Evaluation System")
    st.markdown("""
    ### 🚀 Comprehensive Assignment Assessment
    - Upload your evaluation criteria document
    - Upload the assignment document
    - Get AI-powered, detailed evaluation
    """)

    # Sidebar Configuration
    st.sidebar.header("🔧 Evaluation Configuration")
    
    # File Uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Evaluation Criteria")
        criteria_file = st.file_uploader(
            "Upload Criteria Document", 
            type=['pdf', 'docx'],
            key='criteria_upload'
        )
    
    with col2:
        st.subheader("📄 Assignment Document")
        assignment_file = st.file_uploader(
            "Upload Assignment", 
            type=['pdf', 'docx'],
            key='assignment_upload'
        )

    # AWS and Model Configuration
    col3, col4 = st.columns(2)
    
    with col3:
        region = st.selectbox(
            "🌐 AWS Region", 
            ['us-east-1', 'us-west-2', 'eu-west-1']
        )
    
    with col4:
        model = st.selectbox(
            "🤖 Bedrock Model", 
            [
                'anthropic/claude-v2:1',
                'anthropic/claude-instant-v1', 
                'ai21/j2-ultra', 
                'amazon/titan-text-express-v1'
            ]
        )

    # Evaluation Button
    if st.button("🔍 Evaluate Assignment", use_container_width=True):
        if criteria_file and assignment_file:
            # Parse files based on type
            if criteria_file.type == 'application/pdf':
                criteria_content = FileParser.parse_pdf(criteria_file)
            else:
                criteria_content = FileParser.parse_docx(criteria_file)
            
            if assignment_file.type == 'application/pdf':
                assignment_content = FileParser.parse_pdf(assignment_file)
            else:
                assignment_content = FileParser.parse_docx(assignment_file)

            # Validate parsed content
            if not criteria_content or not assignment_content:
                st.error("Failed to parse files. Please check file contents.")
                st.stop()

            # Initialize Bedrock Evaluator
            evaluator = BedrockEvaluator(
                region_name=region, 
                model_id=model
            )

            # Generate and invoke evaluation
            with st.spinner("🧠 AI Evaluating Assignment..."):
                prompt = evaluator.generate_comprehensive_prompt(
                    assignment_content, 
                    criteria_content
                )
                evaluation = evaluator.invoke_bedrock_model(prompt)

            # Display Results
            st.success("🎉 Evaluation Complete!")

            # Summary Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Score", 
                    evaluation.get('total_score', 'N/A'),
                    help="Overall performance score"
                )
            with col2:
                st.metric("Overall Grade", 
                    evaluation.get('overall_grade', 'N/A'),
                    help="Letter grade representation"
                )
            
            # Detailed Evaluation
            st.subheader("📊 Detailed Evaluation")
            
            # Criterion Breakdown
            if 'criterion_breakdown' in evaluation:
                for criterion in evaluation['criterion_breakdown']:
                    with st.expander(f"{criterion['name']} Evaluation"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Points Awarded", 
                                f"{criterion['awarded_points']}/{criterion['max_points']}")
                        with col2:
                            st.metric("Performance", 
                                f"{(criterion['awarded_points']/criterion['max_points']*100):.1f}%")
                        
                        st.write("**Specific Feedback:**")
                        st.write(criterion['specific_feedback'])
                        
                        st.write("**Improvement Suggestions:**")
                        st.write(criterion['improvement_suggestions'])

            # Overall Insights
            st.subheader("🌟 Overall Insights")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Key Strengths:**")
                for strength in evaluation.get('key_strengths', []):
                    st.markdown(f"- {strength}")
            
            with col2:
                st.markdown("**Areas for Improvement:**")
                for improvement in evaluation.get('areas_for_improvement', []):
                    st.markdown(f"- {improvement}")

            # Textual Overall Feedback
            st.subheader("💡 Comprehensive Feedback")
            st.write(evaluation.get('overall_feedback', 'No detailed feedback available.'))

        else:
            st.warning("Please upload both criteria and assignment documents.")

if __name__ == "__main__":
    main()
