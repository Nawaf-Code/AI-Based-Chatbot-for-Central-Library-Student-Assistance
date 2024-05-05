## Project Overview

The AI-Based Chatbot for Central Library Student Assistance is a virtual assistant built using artificial intelligence (AI) to provide quick and accurate answers to students' queries about the central library.

## Motivation

Traditional methods like personal consultations or static FAQ pages might not always be effective or available to respond to students' inquiries about the central library. This AI-based chatbot was designed to fill this gap, offering a conversational interface for efficient and intelligent answers.

## Objectives
The main objective of this project is to develop a robust AI-based chatbot capable of addressing a wide range of student inquiries efficiently and intelligently.

## Project Setup
### Requirements
- Python 3.x
- `nltk==3.8.1`
- `tensorflow==2.16.1`
- `numpy`

### Installation
1. Clone the repository or download the project files.
2. Install the required libraries using pip:
```bash
pip install -r requirements.txt
```
3. Download the necessary NLTK data packages:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Dataset Structure
Ensure that the `knowledge.json` file follows this structure:
```json
{
  "intents": [
    {
      "tag": "example_tag",
      "patterns": ["Example question 1", "Example question 2"],
      "responses": ["Example response 1", "Example response 2"]
    }
  ]
}
```

## Training the Model
1. Prepare your training data by editing the knowledge.json file.
2. Run `training.py`

## Using the Chatbot
1. Make sure that `chatbotModel.h5`
2. Run `chatbot.py`
