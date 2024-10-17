import os
import dagshub
import mlflow
import pandas as pd
import mlflow.pyfunc
from openai import OpenAI
import textstat
import time
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Load environment variables
api_key = os.getenv("openai_api_key")
dagshub_username = os.getenv("dagshub_username")
dagshub_token = os.getenv("dagshub_token")

# Initialize the DagsHub repo and enable MLflow tracking
dagshub.init(repo_owner=dagshub_username, repo_name='example', mlflow=True)

# Set the tracking URI to your DagsHub repository
mlflow.set_tracking_uri(f"https://dagshub.com/{dagshub_username}/example.mlflow")

# Check if the experiment exists, otherwise create it
if not mlflow.get_experiment_by_name("LLM Evaluation"):
    mlflow.create_experiment("LLM Evaluation")
mlflow.set_experiment("LLM Evaluation")

# Initialize the toxicity classifier and similarity model
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define a custom model class
class OpenAIModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name, system_prompt):
        self.model_name = model_name
        self.system_prompt = system_prompt

    def predict(self, context, model_input):
        client = OpenAI(api_key=api_key)
        responses = []
        for input_text in model_input["inputs"]:
            start_time = time.time()
            response = client.completions.create(
                model=self.model_name,
                prompt=f"{self.system_prompt}\nUser: {input_text}\nAssistant:",
                max_tokens=150,
                temperature=0.7
            )
            latency = time.time() - start_time
            response_text = response.choices[0].text.strip()
            responses.append((response_text, latency))
        return pd.DataFrame(responses, columns=["response", "latency"])

# Define evaluation data
eval_data = pd.DataFrame({
    "inputs": [
        "What is MLflow?",
        "What is Spark?",
    ],
    "ground_truth": [
        "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, a company that specializes in big data and machine learning solutions.",
        "Apache Spark is an open-source, distributed computing system designed for big data processing and analytics. It offers libraries for various tasks such as data ingestion, processing, and analysis.",
    ],
})

# Log the custom OpenAI model
with mlflow.start_run() as run:
    system_prompt = "Answer the following question in two sentences."
    openai_model = OpenAIModelWrapper(model_name="gpt-3.5-turbo-instruct", system_prompt=system_prompt)
    
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=openai_model,
    )

    # Load the model using the python_function flavor
    model_uri = f"runs:/{run.info.run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Custom evaluation loop to capture metrics for each input individually
    individual_results = []
    for idx, row in eval_data.iterrows():
        # Evaluate each prompt individually
        prompt_data = pd.DataFrame([{"inputs": row["inputs"]}])
        response, latency = loaded_model.predict(prompt_data).iloc[0]
        
        # Log the latency as a custom metric
        mlflow.log_metric("latency", latency)
        
        # Compare with ground truth
        ground_truth = row["ground_truth"]
        flesch_kincaid_grade_level = textstat.flesch_kincaid_grade(response)
        ari_grade_level = textstat.automated_readability_index(response)

        # Evaluate toxicity
        toxicity_score = toxicity_classifier(response)[0]['score']
        is_toxic = toxicity_score > 0.5

        # Evaluate similarity with ground truth
        response_embedding = similarity_model.encode(response, convert_to_tensor=True)
        ground_truth_embedding = similarity_model.encode(ground_truth, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(response_embedding, ground_truth_embedding).item()

        # Log additional metrics
        mlflow.log_metric("flesch_kincaid_grade_level", flesch_kincaid_grade_level)
        mlflow.log_metric("ari_grade_level", ari_grade_level)
        mlflow.log_metric("toxicity_score", toxicity_score)
        mlflow.log_metric("similarity_score", similarity_score)

        # Log individual results
        individual_result = {
            "prompt": row["inputs"],
            "response": response,
            "ground_truth": ground_truth,
            "latency": latency,
            "flesch_kincaid_grade_level": flesch_kincaid_grade_level,
            "ari_grade_level": ari_grade_level,
            "toxicity_score": toxicity_score,
            "is_toxic": is_toxic,
            "similarity_score": similarity_score,
        }
        individual_results.append(individual_result)
        
    # Save individual results to a DataFrame
    results_df = pd.DataFrame(individual_results)
    results_df.to_csv('individual_evaluation_results.csv', index=False)
    print(f"See individual evaluation results below: \n{results_df}")
