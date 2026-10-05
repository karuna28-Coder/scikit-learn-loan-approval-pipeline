"""Train and evaluate a complete loan approval preprocessing pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "loan_pipeline.pkl"


def find_dataset() -> Path | None:
	"""Find the CSV at the project root or in the supplied dataset folder."""
	root_dataset = PROJECT_DIR / "loan_approval.csv"
	if root_dataset.is_file():
		return root_dataset

	# The supplied workspace stores the CSV inside a folder with this name.
	if root_dataset.is_dir():
		csv_files = sorted(root_dataset.glob("*.csv"))
		if csv_files:
			return csv_files[0]

	return None


def main() -> None:
	dataset_path = find_dataset()
	if dataset_path is None:
		raise FileNotFoundError(
			"Loan Approval CSV was not found. Place the CSV in the project "
			"folder and name it 'loan_approval.csv'."
		)

	# Pandas reads the CSV into a table called a DataFrame.
	df = pd.read_csv(dataset_path)
	print("Dataset loaded successfully.")
	print(f"Dataset file: {dataset_path}")
	print(f"\nDataset shape: {df.shape}")
	print("\nFirst 5 rows:")
	print(df.head())
	print("\nColumn names:")
	print(df.columns.tolist())
	print("\nMissing values per column:")
	print(df.isnull().sum())

	# Locate Loan_Status without depending on its capitalization.
	target_lookup = {str(column).strip().lower(): column for column in df.columns}
	target_column = target_lookup.get("loan_status")
	if target_column is None:
		raise ValueError(
			"Target column 'Loan_Status' was not found. "
			f"CSV columns are: {df.columns.tolist()}"
		)

	# Rows without a known answer cannot be used for supervised training.
	df = df.dropna(subset=[target_column]).copy()

	# ID values identify records rather than describing loan eligibility.
	id_columns = [
		column
		for column in df.columns
		if str(column).strip().lower() in {"id", "loan_id"}
		or str(column).strip().lower().endswith("_id")
	]
	X = df.drop(columns=[target_column, *id_columns])
	y = df[target_column]

	# Column types determine which preprocessing steps each feature receives.
	numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()
	categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
	if not numerical_columns and not categorical_columns:
		raise ValueError("No usable input feature columns were found in the CSV.")

	print(f"\nTarget column: {target_column}")
	print(f"Removed ID columns: {id_columns}")
	print(f"\nNumerical columns: {numerical_columns}")
	print(f"Categorical columns: {categorical_columns}")

	# Each column group gets its own imputation and transformation steps.
	numerical_preprocessing = make_pipeline(
		SimpleImputer(strategy="median"),
		StandardScaler(),
	)
	categorical_preprocessing = make_pipeline(
		SimpleImputer(strategy="most_frequent"),
		OneHotEncoder(handle_unknown="ignore"),
	)
	preprocessor = ColumnTransformer(
		transformers=[
			("numerical", numerical_preprocessing, numerical_columns),
			("categorical", categorical_preprocessing, categorical_columns),
		]
	)

	# Pipeline ensures training and future predictions use identical processing.
	model_pipeline = Pipeline(
		steps=[
			("preprocessing", preprocessor),
			("classifier", DecisionTreeClassifier(random_state=42)),
		]
	)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)
	print(f"\nTraining samples: {len(X_train)}")
	print(f"Testing samples: {len(X_test)}")

	# Fitting the pipeline learns imputation, scaling, encoding, and the tree.
	model_pipeline.fit(X_train, y_train)
	predictions = model_pipeline.predict(X_test)
	print("\nModel trained successfully.")
	print("The pipeline automatically preprocessed X_test before prediction.")

	# Weighted scores support either binary or multiclass target labels.
	print(f"\nAccuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
	print(
		"Precision (weighted): "
		f"{precision_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
	)
	print(
		"Recall (weighted): "
		f"{recall_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
	)
	print(
		"F1-score (weighted): "
		f"{f1_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
	)
	print("\nClassification Report:")
	print(classification_report(y_test, predictions, zero_division=0))
	print("Confusion Matrix:")
	print(confusion_matrix(y_test, predictions))

	# Save the complete preprocessing-plus-model workflow as one file.
	joblib.dump(model_pipeline, MODEL_PATH)
	print(f"\nPipeline saved successfully: {MODEL_PATH}")

	# Reload the saved file and predict one held-out row as a usage example.
	loaded_pipeline = joblib.load(MODEL_PATH)
	sample_input = X_test.iloc[[0]]
	sample_prediction = loaded_pipeline.predict(sample_input)[0]
	print("\nSaved pipeline sample prediction:")
	print(sample_input)
	print(f"Predicted Loan_Status: {sample_prediction}")


if __name__ == "__main__":
	main()