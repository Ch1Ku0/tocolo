## ToCoLo: A File-level Context-aware Approach for User Review-based Code Localization

We propose \textsc{ToCoLo} (Two-Stage Approach for User Review-based Code Localization), a method that explicitly incorporates file-level context to guide method-level localization. In the first stage, we leverage a large language model (LLM) to generate natural language summaries of code files and use these summaries to identify a small set of candidate files that are most relevant to a given user review. In the second stage, we perform fine-grained method-level matching only within the retrieved files, returning the most relevant code snippets as the final localization results.

The workflow consists of three main stages: Preparation, Fine-tuning, and Inference.

### 1. Preparation Stage

The preparation stage is carried out by running `python generate_file_descriptions.py` and consists of two rounds of interaction with the LLM:

&ensp; 1. First Round: The LLM analyzes the project’s directory structure and generates a global project summary, which serves as stable contextual information for all subsequent file-level analyses. This summary is fixed across all experiments, as it does not directly participate in retrieval and therefore does not affect downstream evaluation.

&ensp; 2. Second Round: The LLM generates descriptions for individual code files, producing semantically rich summaries that will be used for file-level retrieval in later stages.



### 2. Fine-tuning Stage
To fine-tune the model:

&ensp; 1.Execute the training script: 

&ensp; &ensp;  &ensp;  &ensp; `bash train.sh`

&ensp; 2. During fine-tuning, the model uses:

&ensp; &ensp;  &ensp;  &ensp; project_name_train.json for training

&ensp; &ensp;  &ensp;  &ensp; project_name_valid.json for validation

This stage allows the model to learn to effectively map user reviews to code files, preparing for the inference stage where the similarity between user reviews and code is computed, and ranking the method-level snippets within the recalled files.

### 3. Inference Stage

To perform code localization for user reviews:

&ensp; 1. Execute the inference script:

&ensp; &ensp;  &ensp;  &ensp; `bash test.sh`

&ensp; 2. The inference process involves three steps:

&ensp; &ensp; 1. File-Level Recall: Retrieve candidate code files relevant to a given review using the generated descriptions.

&ensp; &ensp; 2. Method Extraction: Collect all method-level code snippets from the recalled files.

&ensp; &ensp; 3. Similarity Calculation: Compute the similarity between the review and each method-level snippet in the recalled files. The similarity scores are used to produce the final method-level localization results.

