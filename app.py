from flask import Flask, jsonify, render_template, request
import PyPDF2
import docx
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import tempfile

def read_pdf(file_path) -> str:
    """Extract text from a PDF file"""
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file_path) -> str:
    """Extract text from a DOCX file"""
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file) -> str:
    """Read text from a file"""
    return file.stream.read().decode('utf-8')

def process_file(uploaded_file) -> str:
    """Process different file formats and return text content"""
    if uploaded_file is None:
        return ""
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.stream.read())
        tmp_file_path = tmp_file.name

    try:
        file_extension = uploaded_file.filename.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            text = read_pdf(tmp_file_path)
        elif file_extension in ['docx', 'doc']:
            text = read_docx(tmp_file_path)
        elif file_extension == 'txt':
            with open(tmp_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = ""
    finally:
        os.unlink(tmp_file_path)
    
    return text

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
def create_vector_db(text: str) -> FAISS:
    """Create FAISS vector database from text"""
   
    
    texts = text_splitter.split_text(text)
    
    
    
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0, 
    groq_api_key='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', 
    model_name="llama-3.1-70b-versatile"
)

def ask_question(query: str, context: str = None) -> str:
    """Get response from Gemini API"""
    if context:
        template = """
        You are a helpful assistant with access to specific context information. Follow these rules to answer:

        1. Respond politely to greetings.
        2. Answer questions based on provided context.
        3. If no context is available, state that explicitly.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        response = llm.predict(prompt.format(context=context, question=query))
    else:
        template = "You are a helpful assistant. Answer naturally:\n\nUser: {question}\nAssistant:"
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"]
        )
        response = llm.predict(prompt.format(question=query))
    
    return response

vectordb = None  # Ensure vectordb is initialized
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/upload_file", methods=["POST"])
def upload_file():
    global vectordb  # Declare as global to modify outside scope
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file is None:
            return jsonify({'response': 'No file uploaded.'})
        
        file_extension = uploaded_file.filename.split('.')[-1].lower()
        supported_extensions = ['pdf', 'docx', 'doc', 'txt']
        if file_extension not in supported_extensions:
            return jsonify({'response': 'Unsupported file format.'})
        
        text = process_file(uploaded_file)
        if len(text) > 2:
            vectordb = create_vector_db(text=text)
            return jsonify({'response': 'File uploaded successfully.'})
        else:
            return jsonify({'response': 'Failed to process the file.'})

@app.route('/ask_question', methods=['POST'])
def answer():
    global vectordb  # Access the global variable
    if vectordb is None:
        return jsonify({'response': 'No document uploaded yet.'})
    
    data = request.get_json()
    question = data.get('question', '')

    results = vectordb.similarity_search_with_score(question, k=2)
    context = "\n".join([doc.page_content for doc, _ in results])
    response = ask_question(question, context)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()
    
