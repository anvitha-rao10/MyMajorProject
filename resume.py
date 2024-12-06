import streamlit as st  # For building the web application
import pandas as pd  # For handling data in DataFrame format
import re  # For working with regular expressions
import nltk  # For natural language processing
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import smtplib  # For sending emails
from email.mime.text import MIMEText  # For creating text-based email messages
from email.mime.multipart import MIMEMultipart  # For creating multipart email messages
from nltk.corpus import stopwords  # For accessing stopwords used in NLP
from nltk.tokenize import word_tokenize  # For tokenizing text into words
from nltk.stem import WordNetLemmatizer  # For lemmatizing words (reducing to base form)
import fitz  # PyMuPDF for reading PDF files
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text into TF-IDF feature vectors
from sklearn.neighbors import NearestNeighbors  # For finding nearest neighbors (used in recommendations or matching)




# Load the cleaned data file into a pandas DataFrame for processing
df = pd.read_csv('cleaned_file.csv')




# Define a set of stop words (common words to exclude during text processing)
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 
    'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 
    'weren', 'won', 'wouldn'
])




# Manually define a simple lemmatization dictionary
# This maps inflected or comparative forms of words to their base form
lemmatizer_dict = {
    'running': 'run', 'ran': 'run', 'runs': 'run',  # Variations of "run"
    'better': 'good', 'best': 'good',  # Comparative and superlative of "good"
    'worse': 'bad', 'worst': 'bad',  # Comparative and superlative of "bad"
    'happier': 'happy', 'happiest': 'happy',  # Variations of "happy"
    'sadder': 'sad', 'saddest': 'sad',  # Variations of "sad"
    'more': 'much', 'most': 'much',  # Comparative and superlative of "much"
    'less': 'little', 'least': 'little',  # Variations of "little"
    'doing': 'do', 'did': 'do', 'does': 'do', 'done': 'do'  # Variations of "do"
}




# Function to clean and preprocess text
def clean_text(txt):
    # Remove URLs, mentions, hashtags, non-ASCII characters, and unwanted symbols
    clean_text = re.sub(r'http\S+\s|RT|cc|#\S+\s|@\S+|[^\x00-\x7f]', ' ', txt)
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip().lower()  # Normalize spacing and lowercase
    
    # Tokenization: Split text into individual words
    tokens = re.findall(r'\b\w+\b', clean_text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization using the manually defined dictionary
    tokens = [lemmatizer_dict.get(word, word) for word in tokens]
    return ' '.join(tokens)  # Return cleaned text as a single string

# Function to extract text content from uploaded PDF files using PyMuPDF
def extract_text_from_pdf(uploaded_file):
    text = ""  # Initialize empty string to hold extracted text
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:  # Open the PDF
            for page in doc:
                text += page.get_text()  # Extract text from each page
    except Exception as e:
        st.error(f"Failed to extract text from the PDF: {e}")  # Handle errors gracefully
    return text

# Initialize the TF-IDF vectorizer and fit the job descriptions data
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
job_descriptions = df['Skills'].apply(clean_text).tolist()  # Clean the job descriptions
X = vectorizer.fit_transform(job_descriptions)  # Transform text into TF-IDF feature vectors

# Initialize and train the K-Nearest Neighbors model
knn = NearestNeighbors(n_neighbors=5, metric='cosine')  # Use cosine similarity for matching
knn.fit(X)  # Fit the model with the transformed data

# Add CSS styles for customizing the app's appearance
st.markdown("""<style>
    body {
        background-color: grey;  /* Set background color */
        color: pink;  /* Default text color */
        font-family: Georgia, 'Times New Roman', Times, serif;
    }

    .title {
        text-align: center;
        color: blue;  /* Title color */
        font-size: 30px;
        font-weight: 700;
        text-transform: uppercase;
    }

    .subtitle {
        text-align: left;
        font-size: 20px;
        color: red;  /* Subtitle color */
        font-weight: 650;
        font-family: Arial, Helvetica, sans-serif;
    }

    .footer {
        text-align: center;
        padding: 20px;
        color: white;
        background-color: #1d61b4;  /* Footer background */
        font-size: 14px;
    }

    .job-list {
        display: grid;
        grid-template-columns: repeat(3, 1fr);  /* Display jobs in 3 columns */
        gap: 20px;
        margin-top: 20px;
    }

    .job-item {
        background-color: #ff6f61;  /* Job card background */
        color: white;
        border-radius: 15px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        box-sizing: border-box;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  /* Add shadow for elevation */
        padding: 20px;
    }

    .job-item:hover {
        transform: scale(1.05);  /* Scale up slightly on hover */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);  /* Add shadow on hover */
        background-color: #ff4b39;  /* Darker coral on hover */
    }

    .load-more-btn {
        background-color: #64b5f6;  /* Button color */
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 50px;
        text-align: center;
        cursor: pointer;
        border: none;
        display: block;
        margin-left: auto;
        margin-right: auto;
        animation: pulse 1s infinite;  /* Add pulsing animation */
        padding: 15px 30px;
    }

    .load-more-btn:hover {
        background-color: #039be5;  /* Change color on hover */
        transform: scale(1.05);
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }

    table {
        width: 100%;  /* Table spans full width */
        text-align: center;
        border-collapse: collapse;  /* Remove spacing between cells */
        margin-top: 20px;
    }

    table, th, td {
        border: 1px solid #5e92f3;  /* Add borders */
        border-radius: 5px;
    }

    th, td {
        padding: 12px;
        background-color: #f3f9fc;  /* Light blue cell background */
    }

    th {
        background-color: #1976d2;  /* Header row color */
        color: white;
    }

    .btn-primary {
        background-color: #4caf50;  /* Green button */
    }

    .btn-primary:hover {
        background-color: #388e3c;  /* Darker green on hover */
    }
</style>
""", unsafe_allow_html=True)  # Allow unsafe HTML for custom CSS




# Extract job titles and skills from the CSV file
job_titles = df['Job Title'].sort_values().unique()  # Extract unique job titles in alphabetical order
skills_dict = dict(zip(df['Job Title'], df['Skills']))  # Create a dictionary mapping job titles to required skills

# Sidebar Navigation
st.sidebar.title("Navigation")  # Title for the sidebar
page = st.sidebar.selectbox("Go to", ["About Us", "Resume Analyzer", "Find Jobs", "Enhance Skills", "Contact Us"])  # Navigation menu

# Header (no line breaks, ensures single-line heading)
st.markdown("<div class='title'>Intelligent Resume Analysis And Job Fit Assessment System</div>", unsafe_allow_html=True)







# About Us Page
if page == "About Us":
    st.markdown("<div class='subtitle'>About Us</div>", unsafe_allow_html=True)  # Subtitle
    st.write("""
    Welcome to the **Intelligent Resume Analysis and Job Fit Assessment System**! 
    Our platform is designed to leverage **Artificial Intelligence** to:
    - Match your resumes with the most relevant job descriptions.
    - Help you discover job opportunities tailored to your skillset.
    - Provide actionable recommendations to enhance your skills.
    
    We aim to make the job search and resume analysis process seamless, accurate, and empowering for job seekers worldwide.
    """)  # Description of the system

    # Job Search Dropdown with Skills Display
    st.markdown("<div class='subtitle'>Search for Job Titles</div>", unsafe_allow_html=True)  # Subtitle for job search
    search_job_title = st.selectbox("Search Job Title", job_titles)  # Dropdown for selecting job titles
    if search_job_title:
        st.markdown(f"<div style='margin-left: 20px;'><b>Skills Required for {search_job_title}:</b> {skills_dict[search_job_title]}</div>", unsafe_allow_html=True)

    # Track the index of displayed jobs in session state
    if 'job_index' not in st.session_state:
        st.session_state.job_index = 0  # Initialize the job index to start from the first job
        st.session_state.job_list = []  # Initialize an empty list to store displayed jobs

    # Show previously displayed jobs
    st.markdown("<div class='subtitle'>Explore Job Titles</div>", unsafe_allow_html=True)  # Subtitle for job exploration
    for job_title in st.session_state.job_list:  # Loop through and display previously selected jobs
        st.markdown(f"<div style='margin-left: 20px;'><b>{job_title}:</b> {skills_dict[job_title]}</div>", unsafe_allow_html=True)

    # Display jobs in a clickable grid
    end_index = min(st.session_state.job_index + 10, len(job_titles))  # Determine the subset of jobs to display
    job_subset = job_titles[st.session_state.job_index:end_index]  # Extract a subset of job titles

    col_count = 5  # Number of job titles to display per row
    for i in range(0, len(job_subset), col_count):  # Loop through job titles in chunks of 'col_count'
        cols = st.columns(col_count)  # Create columns for job titles
        for j in range(col_count):
            if i + j < len(job_subset):  # Ensure index is within bounds
                job_title = job_subset[i + j]  # Get the job title
                with cols[j]:  # Place the button in the respective column
                    if st.button(job_title):  # Create a button for the job title
                        st.markdown(f"<div style='margin-left: 20px;'><b>Skills Required for {job_title}:</b> {skills_dict[job_title]}</div>", unsafe_allow_html=True)
                        # Add the clicked job to the displayed list
                        if job_title not in st.session_state.job_list:
                            st.session_state.job_list.append(job_title)

    # "Explore more jobs" button with animation
    if end_index < len(job_titles):  # Check if there are more jobs to display
        explore_button = st.button("Click here to explore more jobs", key="explore_more", use_container_width=True)
        if explore_button:
            st.session_state.job_index = end_index  # Update index to load the next set of jobs








# Resume Analyzer Page
elif page == "Resume Analyzer":
    st.markdown("<div class='subtitle'>Resume Analyzer</div>", unsafe_allow_html=True)  # Subtitle for resume analyzer
    uploaded_file = st.file_uploader("Upload your resume PDF", type="pdf")  # File uploader for resume PDFs

    if uploaded_file:
        with st.spinner("Processing resume..."):  # Show a spinner while processing
            resume_text = extract_text_from_pdf(uploaded_file)  # Extract text from the uploaded PDF
            cleaned_resume = clean_text(resume_text)  # Clean the extracted text

            # Check for compulsory resume words
            compulsory_words = ["skill"]
            if not any(word.lower() in cleaned_resume for word in compulsory_words):  # Validate if the file is a resume
                st.error("This does not appear to be a valid resume. Please upload a valid resume PDF.")
            else:
                # Vectorize resume text
                resume_vector = vectorizer.transform([cleaned_resume])  # Transform the resume text into vectors

                # Find the Top 5 Matching Jobs
                distances, indices = knn.kneighbors(resume_vector)  # Get the nearest neighbors for the resume

                # Ensure we're always getting the top 5 jobs, even if fewer are found
                num_jobs = min(5, len(distances[0]))  # Use min to avoid index error

                # Check if the number of indices is less than expected
                top_5_jobs = df.iloc[indices[0][:num_jobs]]  # Get job details for the top matches
                accuracy_scores = []

                # Display the top jobs and calculate accuracy
                st.markdown("<div class='subtitle'>Top Matching Job Titles</div>", unsafe_allow_html=True)

                # Use neutral style for boxes
                for i in range(num_jobs):  # Loop through the top matches
                    job_index = indices[0][i]  # Get the job index
                    score = 1 - distances[0][i]  # Calculate accuracy (1 - distance gives similarity score)
                    accuracy_scores.append(score)
                    job_row = df.iloc[job_index]  # Get job details

                    # Box styling with neutral background
                    st.markdown(f"""
                    <div style="
                        padding: 20px;
                        margin: 10px;
                        border-radius: 10px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        font-size: 1.1em;
                    ">
                        <strong>Job Title:</strong> {job_row['Job Title']}<br>
                        <strong>Matched Skills:</strong> {job_row['Skills']}<br>
                        <strong>Accuracy:</strong> {score:.2f}
                    </div>
                    """, unsafe_allow_html=True)

                # Pie chart visualization for the top job accuracy scores
                labels = top_5_jobs['Job Title']  # Labels for the pie chart
                sizes = [score * 100 for score in accuracy_scores]  # Convert accuracy scores to percentages
                colors = plt.cm.Paired.colors  # Color palette

                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)  # Create the pie chart
                ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle

                st.pyplot(fig)  # Display the pie chart

                # Highlight and animate the top job's name
                top_job_name = top_5_jobs.iloc[0]['Job Title']  # Get the top matching job

                # CSS animation and styling for highlighting
                st.markdown(f"""
                <div style="
                    font-size: 2em;
                    font-weight: bold;
                    color: #ff6347;  /* Tomato color for emphasis */
                    text-align: center;
                    animation: pulse 2s infinite;
                ">
                    Top Matching Job: <span style="color: #008080;">{top_job_name}</span>
                </div>

                <style>
                    @keyframes pulse {{
                        0% {{ transform: scale(1); }}
                        50% {{ transform: scale(1.1); }}
                        100% {{ transform: scale(1); }}
                    }}
                </style>
                """, unsafe_allow_html=True)

                # Add encouraging message
                st.markdown("""
                <div class='subtitle' style="color:green;">Keep it up! You're on the right track to finding your dream job!</div>
                <p style="text-align:center;">By analyzing your resume, we've matched you with top roles based on your skills fit. Keep enhancing your skills and applying for opportunities!</p>
                """, unsafe_allow_html=True)
    





    # Create 3 columns for the job portal links
col1, col2, col3 = st.columns(3)

# Add job portal links to the respective columns
with col1:
    # Unstop - Job portal for various fields
    st.markdown('<a href="https://www.unstop.com/jobs" target="_blank" class="job-portal-link">Find Jobs on Unstop</a>', unsafe_allow_html=True)
    # WorkIndia - Job portal for local and entry-level jobs
    st.markdown('<a href="https://www.workindia.in/jobs" target="_blank" class="job-portal-link">Find Jobs on WorkIndia</a>', unsafe_allow_html=True)
    # Internshala - Platform for internships
    st.markdown('<a href="https://www.internshala.com/internships" target="_blank" class="job-portal-link">Find Internships on Internshala</a>', unsafe_allow_html=True)

with col2:
    # LinkedIn - Professional networking and job platform
    st.markdown('<a href="https://www.linkedin.com/jobs" target="_blank" class="job-portal-link">Find Jobs on LinkedIn</a>', unsafe_allow_html=True)
    # Glassdoor - Job platform with reviews and salary insights
    st.markdown('<a href="https://www.glassdoor.com/Job/index.htm" target="_blank" class="job-portal-link">Find Jobs on Glassdoor</a>', unsafe_allow_html=True)
    # Indeed - Popular job search platform
    st.markdown('<a href="https://www.indeed.com" target="_blank" class="job-portal-link">Find Jobs on Indeed</a>', unsafe_allow_html=True)

with col3:
    # Naukri - Job portal for India
    st.markdown('<a href="https://www.naukri.com" target="_blank" class="job-portal-link">Find Jobs on Naukri</a>', unsafe_allow_html=True)
    # AngelList - Job portal for startups
    st.markdown('<a href="https://www.angel.co/jobs" target="_blank" class="job-portal-link">Find Jobs on AngelList</a>', unsafe_allow_html=True)
    # SimplyHired - General job search platform
    st.markdown('<a href="https://www.simplyhired.com" target="_blank" class="job-portal-link">Find Jobs on SimplyHired</a>', unsafe_allow_html=True)

# Display motivational tips for job seekers
st.markdown("<div class='encouragement'>Pro Tip: While applying, make sure to tailor your resume for each job. Highlight relevant skills, experiences, and achievements that align with the job description.</div>", unsafe_allow_html=True)
st.markdown("<div class='encouragement'>Keep improving your skills and learning new ones. The right job is just around the corner!</div>", unsafe_allow_html=True)






# Enhance Skills Page
if page == "Enhance Skills":
    # Introduction for skill enhancement
    st.markdown("<div ><h3 class='subtitle'>Enhance Your Skills for Success</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='content'>
        <p>Continuous learning and skill development are essential for staying competitive in the job market. Here, you can find a variety of resources to help you improve your skills, whether you're looking to boost your technical abilities or enhance your soft skills.</p>
    </div>
    """, unsafe_allow_html=True)

    # Top Skills to Learn
    st.markdown("<div ><h3 class='subtitle'>Top Skills to Learn</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    <ul>
        <li><strong>Technical Skills:</strong> Data Analysis, Programming (Python, JavaScript), Machine Learning, Cloud Computing</li>
        <li><strong>Soft Skills:</strong> Communication, Leadership, Problem-Solving, Time Management</li>
        <li><strong>Industry-Specific Skills:</strong> Digital Marketing, UX/UI Design, Financial Analysis, Business Development</li>
    </ul>
    """, unsafe_allow_html=True)

    # Create 3 columns for online course platforms
    col1, col2, col3 = st.columns(3)

    # Add links to online course platforms
    with col1:
        # Coursera - Platform for structured courses
        st.markdown('<a href="https://www.coursera.org" target="_blank" class="skill-button">Coursera </a>', unsafe_allow_html=True)
        # edX - Online platform for academic courses
        st.markdown('<a href="https://www.edx.org" target="_blank" class="skill-button">edX</a>', unsafe_allow_html=True)
        # Udemy - Learning platform with diverse courses
        st.markdown('<a href="https://www.udemy.com" target="_blank" class="skill-button">Udemy</a>', unsafe_allow_html=True)

    with col2:
        # LinkedIn Learning - Professional skill development
        st.markdown('<a href="https://www.linkedin.com/learning" target="_blank" class="skill-button">LinkedIn</a>', unsafe_allow_html=True)
        # Skillshare - Creative and business skills platform
        st.markdown('<a href="https://www.skillshare.com" target="_blank" class="skill-button">Skillshare </a>', unsafe_allow_html=True)
        # FutureLearn - Online courses from top universities
        st.markdown('<a href="https://www.futurelearn.com" target="_blank" class="skill-button">FutureLearn</a>', unsafe_allow_html=True)

    with col3:
        # Codecademy - Programming and coding skills
        st.markdown('<a href="https://www.codecademy.com" target="_blank" class="skill-button">Codecademy </a>', unsafe_allow_html=True)
        # LeetCode - Coding challenges and algorithms
        st.markdown('<a href="https://www.leetcode.com" target="_blank" class="skill-button">LeetCode </a>', unsafe_allow_html=True)
        # Khan Academy - General education platform
        st.markdown('<a href="https://www.khanacademy.org" target="_blank" class="skill-button">Khan Academy</a>', unsafe_allow_html=True)

# Add custom CSS styles for hover effects and button appearance
st.markdown("""
<style>
/* Button styling for skill-related links */
.skill-button {
    color: #1e3a8a;  /* Dark blue for text */
    padding: 15px 32px;  /* Spacing for the button */
    border-radius: 8px;  /* Rounded corners */
    border: 2px solid #b0bec5;  /* Light border */
    transition: background-color 0.3s ease, color 0.3s ease;  /* Smooth hover effect */
}

/* Change button appearance on hover */
.skill-button:hover {
    background-color: #ffffff;  /* White background */
    color: #1e3a8a;  /* Dark blue text */
    border: 2px solid #1e3a8a;  /* Dark border */
}

/* General content text styling */
.content {
    font-size: 18px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)







# Streamlit page for Contact Us
if page == "Contact Us":
    # Display a subtitle for the Contact Us page
    st.markdown("<div class='subtitle'>Contact Us</div>", unsafe_allow_html=True)

    # Description for the contact form
    st.write("""
    **We'd love to hear from you!**
    If you have any questions, feedback, or need assistance, feel free to reach out to us.
    You can contact us using the following methods:

    - **Email**: [resumeanalyzerr@gmail.com](mailto:resumeanalyzerr@gmail.com)
    - **Phone**: +91 9480199605
    """)

    # Custom CSS for better design and layout
    st.markdown("""
    <style>
    /* Style for the submit button */
    .stButton>button {
        background-color: #4CAF50;  /* Green background */
        color: white;  /* White text */
        font-size: 16px;  /* Text size */
        border: none;  /* Remove border */
        cursor: pointer;  /* Pointer cursor on hover */
        padding: 10px 20px;  /* Padding for button */
        border-radius: 5px;  /* Rounded corners */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
    }

    /* Hover effect for the submit button */
    .stButton>button:hover {
        background-color: #45a049;  /* Darker green on hover */
    }

    /* Style for input fields (text) */
    .stTextInput>div>input {
        font-size: 16px;  /* Text size */
        padding: 10px;  /* Padding inside input */
        border-radius: 5px;  /* Rounded corners */
        border: 1px solid #ccc;  /* Light gray border */
    }

    /* Style for textarea input */
    .stTextArea>div>textarea {
        font-size: 16px;  /* Text size */
        padding: 10px;  /* Padding inside textarea */
        border-radius: 5px;  /* Rounded corners */
        border: 1px solid #ccc;  /* Light gray border */
    }

    /* Add margin between input fields */
    .stTextInput, .stTextArea {
        margin-bottom: 20px;  /* Space below fields */
    }

    /* Style for rating stars (gold color) */
    .stRadio>div>label>div {
        display: flex;  /* Flex layout for alignment */
        justify-content: center;  /* Center-align stars */
        font-size: 24px;  /* Size of stars */
        color: #FFD700;  /* Gold color for stars */
    }

    /* Add pointer cursor for stars (radio inputs) */
    .stRadio>div>label>div>input {
        cursor: pointer;  /* Pointer cursor */
    }
    </style>
    """, unsafe_allow_html=True)

# Footer section
st.markdown("<div class='footer'>© 2024 Resume Analyzer (ARKK)</div>", unsafe_allow_html=True)
