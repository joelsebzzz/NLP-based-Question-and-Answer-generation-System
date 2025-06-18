import os
import re
import torch
from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import PyPDF2
from pyngrok import ngrok  # ngrok for exposing local server

import config  # Your config file
from inference_utils import parse_qna_output  # Re-use the parser

app = Flask(__name__)

# --- Model Loading ---
tokenizer_short_flask = None
model_short_flask = None
tokenizer_long_flask = None
model_long_flask = None
device_flask = None


def load_models_for_flask_app():
    global tokenizer_short_flask, model_short_flask, tokenizer_long_flask, model_long_flask, device_flask

    print("Flask App: Loading models...")
    device_flask = config.DEVICE  # Use device from config
    print(f"Flask App: Using device: {device_flask}")

    # Load Short Q&A Model
    if os.path.exists(config.SHORT_QNA_MODEL_LOAD_PATH):
        print(f"Flask App: Loading Short Q&A model from: {config.SHORT_QNA_MODEL_LOAD_PATH}")
        try:
            tokenizer_short_flask = AutoTokenizer.from_pretrained(config.SHORT_QNA_MODEL_LOAD_PATH)
            model_short_flask = AutoModelForSeq2SeqLM.from_pretrained(config.SHORT_QNA_MODEL_LOAD_PATH)
            model_short_flask.to(device_flask)
            model_short_flask.eval()
            print("Flask App: Short Q&A model loaded.")
        except Exception as e:
            print(f"Flask App: Error loading Short Q&A model: {e}")
            model_short_flask = None
    else:
        print(f"Flask App Warning: Short Q&A model path not found: {config.SHORT_QNA_MODEL_LOAD_PATH}")

    # Load Long Q&A Model
    if os.path.exists(config.LONG_QNA_MODEL_LOAD_PATH):
        print(f"Flask App: Loading Long Q&A model from: {config.LONG_QNA_MODEL_LOAD_PATH}")
        try:
            tokenizer_long_flask = AutoTokenizer.from_pretrained(config.LONG_QNA_MODEL_LOAD_PATH)
            model_long_flask = AutoModelForSeq2SeqLM.from_pretrained(config.LONG_QNA_MODEL_LOAD_PATH)
            model_long_flask.to(device_flask)
            model_long_flask.eval()
            print("Flask App: Long Q&A model loaded.")
        except Exception as e:
            print(f"Flask App: Error loading Long Q&A model: {e}")
            model_long_flask = None
    else:
        print(f"Flask App Warning: Long Q&A model path not found: {config.LONG_QNA_MODEL_LOAD_PATH}")

    print("Flask App: Model loading attempt complete.")


def generate_multiple_qna_flask(context, answer_type="short"):
    qna_list, error_message = [], None

    model_to_use, tokenizer_to_use = None, None
    gen_params = {}

    if answer_type == "short":
        if not model_short_flask or not tokenizer_short_flask:
            return [], "Short Q&A model not loaded."
        model_to_use, tokenizer_to_use = model_short_flask, tokenizer_short_flask
        gen_params = {
            "prefix": config.INPUT_PREFIX_SHORT, "max_in": config.MAX_INPUT_LENGTH_SHORT,
            "max_out": config.INF_MAX_OUTPUT_LENGTH_SHORT,  # Use the general inference max length
            "beams": config.FLASK_INF_NUM_BEAMS_SHORT, "early_stop": config.FLASK_INF_EARLY_STOPPING_SHORT,
            "no_repeat": config.FLASK_INF_NO_REPEAT_NGRAM_SIZE_SHORT,
            "num_seq": config.FLASK_NUM_RETURN_SEQUENCES_SHORT,
            "temp": config.FLASK_TEMPERATURE_SHORT, "top_p": config.FLASK_TOP_P_SHORT, "top_k": config.FLASK_TOP_K_SHORT
        }
    elif answer_type == "long":
        if not model_long_flask or not tokenizer_long_flask:
            return [], "Long Q&A model not loaded."
        model_to_use, tokenizer_to_use = model_long_flask, tokenizer_long_flask
        gen_params = {
            "prefix": config.INPUT_PREFIX_LONG, "max_in": config.MAX_INPUT_LENGTH_LONG,
            "max_out": config.INF_MAX_OUTPUT_LENGTH_LONG,  # Use the general inference max length
            "beams": config.FLASK_INF_NUM_BEAMS_LONG, "early_stop": config.FLASK_INF_EARLY_STOPPING_LONG,
            "no_repeat": config.FLASK_INF_NO_REPEAT_NGRAM_SIZE_LONG,
            "num_seq": config.FLASK_NUM_RETURN_SEQUENCES_LONG,
            "temp": config.FLASK_TEMPERATURE_LONG, "top_p": config.FLASK_TOP_P_LONG, "top_k": config.FLASK_TOP_K_LONG
        }
    else:
        return [], "Invalid answer type specified."

    if not context or not isinstance(context, str):
        return [], "Invalid context provided."

    input_text = f"{gen_params['prefix']}{context.strip()}"
    try:
        inputs = tokenizer_to_use(input_text, max_length=gen_params['max_in'], padding="max_length", truncation=True,
                                  return_tensors="pt")
    except Exception as e:
        return [], f"Error during tokenization: {str(e)}"

    input_ids = inputs.input_ids.to(device_flask)
    attention_mask = inputs.attention_mask.to(device_flask)

    print(
        f"Flask App: Generating {gen_params['num_seq']} {answer_type} Q&A pairs with temp={gen_params['temp']}, top_p={gen_params['top_p']}, top_k={gen_params['top_k']}, beams={gen_params['beams']}...")

    try:
        with torch.no_grad():
            generation_args = {
                "input_ids": input_ids, "attention_mask": attention_mask,
                "max_length": gen_params['max_out'],
                "num_return_sequences": gen_params['num_seq'],
                "no_repeat_ngram_size": gen_params['no_repeat'],
                "do_sample": True,  # Crucial for temperature, top_k, top_p
                "temperature": gen_params['temp'],
                "top_p": gen_params['top_p'],
                "top_k": gen_params['top_k']
            }
            # Only include num_beams and early_stopping if num_beams > 1
            # Note: For sampling (do_sample=True), num_beams=1 is typical.
            # If num_beams > 1, it switches to beam search, and temp/top_k/top_p might behave differently or be ignored.
            if gen_params['beams'] > 1:
                generation_args["num_beams"] = gen_params['beams']
                generation_args["early_stopping"] = gen_params['early_stop']
                # Beam search usually doesn't use do_sample, temp, top_k, top_p in the same way.
                # The Hugging Face generate function handles this, but be aware of the interaction.
                # For pure sampling, ensure beams=1.
                del generation_args["do_sample"]  # if beams > 1, sampling params are less relevant
                del generation_args["temperature"]
                del generation_args["top_p"]
                del generation_args["top_k"]

            outputs = model_to_use.generate(**generation_args)
    except Exception as e:
        return [], f"Error during model generation: {str(e)}"

    for i, output_sequence in enumerate(outputs):
        generated_text = tokenizer_to_use.decode(output_sequence, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
        print(f"Flask App: Raw generated output #{i + 1} ({answer_type}): {generated_text}")
        parsed_qna = parse_qna_output(generated_text)  # Use the shared parser
        if parsed_qna:
            qna_list.append(parsed_qna)

    if not qna_list and gen_params['num_seq'] > 0:
        error_message = f"Could not parse any valid Q&A pairs from the generated {answer_type} output."

    # Simple de-duplication based on question
    if qna_list:
        unique_qna_list = []
        seen_questions = set()
        for qna_pair_item in qna_list:
            question_key = qna_pair_item["question"].lower().strip()
            if question_key not in seen_questions:
                unique_qna_list.append(qna_pair_item)
                seen_questions.add(question_key)
        if len(unique_qna_list) < len(qna_list):
            print(f"Flask App: De-duplicated Q&A pairs. Original: {len(qna_list)}, Unique: {len(unique_qna_list)}")
        qna_list = unique_qna_list

    return qna_list, error_message


# --- File Processing ---
def extract_text_from_txt(file_stream):
    try:
        return file_stream.read().decode('utf-8')
    except Exception as e:
        print(f"Error reading txt: {e}")
        return None


def extract_text_from_pdf(file_stream):
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = "".join(page.extract_text() or "" for page in reader.pages if page.extract_text())
        return text.strip() if text else ""  # Ensure empty string if no text, not None
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    error_message, passage_input_display, answer_type_selected, qna_results_display = None, "", "short", []

    if request.method == 'POST':
        passage_text_form = request.form.get('passage_text', '').strip()
        uploaded_file = request.files.get('file')
        answer_type_selected = request.form.get('answer_type', 'short')

        context_to_process = ""

        if uploaded_file and uploaded_file.filename != '':
            filename = uploaded_file.filename
            if filename.endswith('.txt'):
                context_to_process = extract_text_from_txt(uploaded_file.stream)
            elif filename.endswith('.pdf'):
                context_to_process = extract_text_from_pdf(uploaded_file.stream)
            else:
                error_message = "Invalid file type. Please upload a .txt or .pdf file."

            if context_to_process is None and not error_message:  # Error during extraction
                error_message = "Could not extract text from the uploaded file."
            elif context_to_process == "" and not error_message:  # File was empty or extraction yielded nothing
                error_message = "The uploaded file is empty or contains no extractable text."

            # For display, show snippet or filename
            passage_input_display = context_to_process[:1000] + "..." if context_to_process and len(context_to_process) > 1000 else (context_to_process or f"File: {filename} (processing failed or empty)")

        elif passage_text_form:
            context_to_process = passage_text_form
            passage_input_display = passage_text_form  # Display what user typed
        else:  # No input provided
            error_message = "Please enter a passage or upload a file."

        if context_to_process and not error_message:  # If we have context and no prior errors
            word_count = len(context_to_process.split())
            if word_count == 0:  # Context became empty after stripping or was just whitespace
                error_message = "The provided text is empty after processing."
            elif word_count > config.FLASK_WORD_LIMIT:
                error_message = f"Passage exceeds word limit of {config.FLASK_WORD_LIMIT} words (found {word_count}). Please shorten it."
            else:
                # Check if the required model is loaded
                model_ready = (answer_type_selected == "short" and model_short_flask and tokenizer_short_flask) or \
                              (answer_type_selected == "long" and model_long_flask and tokenizer_long_flask)

                if not model_ready:
                    error_message = f"The model for '{answer_type_selected}' answers is not loaded. Please check server logs."
                else:
                    qna_results_display, gen_error = generate_multiple_qna_flask(context_to_process, answer_type_selected)
                    if gen_error:
                        error_message = gen_error
                    elif not qna_results_display:  # No error, but no results
                        error_message = f"No valid Q&A pairs were generated for the '{answer_type_selected}' type. The input might be too short or lack clear question-answer content."

        # If context_to_process is still empty/None after checks and there's no specific error, set a generic one.
        elif not context_to_process and not passage_text_form and not (
                uploaded_file and uploaded_file.filename != '') and not error_message:
            error_message = "Please enter a passage or upload a file."
        elif not context_to_process and passage_text_form and not error_message:  # User typed only whitespace
            error_message = "Please enter a non-empty passage."
            passage_input_display = passage_text_form

    return render_template('index.html',
                           error=error_message,
                           passage_input=passage_input_display,
                           answer_type_input=answer_type_selected,
                           qna_results=qna_results_display,
                           WORD_LIMIT=config.FLASK_WORD_LIMIT)


if __name__ == '__main__':
    # Ensure the templates directory exists (it should if index.html is there)
    if not os.path.exists("templates"):
        os.makedirs("templates")
        print("Created templates directory. Make sure index.html is inside it.")

    # Check if index.html exists
    if not os.path.exists("templates/index.html"):
        print("ERROR: templates/index.html not found! Please create it.")
        print("You can copy the HTML content from the Jupyter Notebook into templates/index.html.")
        exit()

    load_models_for_flask_app()  # Load models at startup

    # Setup ngrok (replace with your actual authtoken if needed, or set via environment variable)
    # It's better to set this via environment variable `NGROK_AUTHTOKEN`
    # or by running `ngrok config add-authtoken <YOUR_TOKEN>` once in your terminal.
    # If NGROK_AUTHTOKEN is set in config.py, use it, otherwise, ngrok will try to use the default config.
    # ngrok_auth_token = getattr(config, 'NGROK_AUTHTOKEN', None)
    # if ngrok_auth_token:
    #     ngrok.set_auth_token(ngrok_auth_token)
    # else:
    # print("NGROK_AUTHTOKEN not found in config.py. ngrok will use default configuration if available.")
    # print("If you haven't configured ngrok CLI with an authtoken, tunnels might be temporary or restricted.")
    # print("Consider running: ngrok config add-authtoken YOUR_TOKEN")

    # The ngrok authtoken from your notebook:
    # IMPORTANT: THIS IS A SECURITY RISK TO HARDCODE.
    # It's better to run `!ngrok config add-authtoken YOUR_TOKEN` in your environment
    # or set an environment variable.
    try:
        ngrok.set_auth_token("2xkc0M26NfumNL9ASRxPB1ji1Ak_4J9UXseie1BgEDBUCMUc1")  # Token from your notebook
        public_url = ngrok.connect(5000)
        print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")
    except Exception as e:
        print(f"Could not start ngrok: {e}. Running locally only.")
        public_url = None

    app.run(port=5000, use_reloader=False)  # use_reloader=False is important with ngrok and model loading