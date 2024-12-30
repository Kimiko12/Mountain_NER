"""
Script for preparing NER datasets:
1. Data retrieved from Hugging Face (via API).
2. Locally downloaded data from Kaggle (or other sources).

Combines them into a single dataset in IOB format.
"""

import os
import re
import ast
import logging
import requests
import pandas as pd
from typing import Tuple, List
from dotenv import load_dotenv


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the script.

    Parameters
    ----------
    level : int
        Logging level, defaults to logging.INFO.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


class DataPreparation:
    """
    A class to prepare two datasets for model training:
    1. Data from Hugging Face (accessed via API).
    2. Data from a local Kaggle (or other) source, pre-downloaded.

    The datasets are converted into the IOB (Inside-Outside-Begin) labeling format for NER tasks.
    """

    def __init__(
        self,
        data_path_to_first_source: str,
        base_url: str,
        dataset_name: str = "telord/mountains-ner-dataset",
        config: str = "default",
        split: str = "train",
        length: int = 100
    ) -> None:
        """
        Initialize the data preparation object.

        Parameters
        ----------
        data_path_to_first_source : str
            Path to the local CSV dataset (with 'text' and 'marker' columns).
        base_url : str
            Base URL for the Hugging Face dataset server (e.g., 'https://datasets-server.huggingface.co/rows').
        dataset_name : str, optional
            The name of the Hugging Face dataset, by default "telord/mountains-ner-dataset".
        config : str, optional
            Configuration name of the HF dataset, by default "default".
        split : str, optional
            Split name to retrieve (e.g., 'train', 'validation'), by default "train".
        length : int, optional
            Batch size for pagination (number of rows to retrieve at once from the API), by default 100.

        Raises
        ------
        ValueError
            If the environment variable HUGGING_FACE_API_KEY is not set or is empty.
        """
        self.data_path_to_first_source = data_path_to_first_source
        self.base_url = base_url
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.length = length

        # Load environment variables and check for the API key
        load_dotenv()
        self.API_KEY = os.getenv('HUGGING_FACE_API_KEY')
        if not self.API_KEY:
            raise ValueError("The environment variable HUGGING_FACE_API_KEY is not set or is empty.")

    @staticmethod
    def _clean_list_string(val: str) -> List[str]:
        """
        Clean and parse a string representing a list of tokens/labels into an actual list of strings.

        Parameters
        ----------
        val : str
            String to be cleaned and parsed.

        Returns
        -------
        List[str]
            A list of strings if parsing is successful, otherwise an empty list.
        """
        # If already a list, just return
        if isinstance(val, list):
            return val
        
        # If it's not a string, return an empty list
        if not isinstance(val, str):
            return []

        val = val.strip()

        # Remove single quotes
        val = val.replace("'", "")

        # Fix comma issues
        val = re.sub(r",\s*\]", "]", val)
        val = re.sub(r",\s*\)", ")", val)

        # Ensure brackets
        if not val.startswith("["):
            val = "[" + val
        if not val.endswith("]"):
            val = val + "]"

        # Safely evaluate string to list
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return []
        except (SyntaxError, ValueError) as e:
            logging.warning(f"Failed to parse string: {val}. Error: {e}")
            return []

    def get_data_from_huggingface(self) -> pd.DataFrame:
        """
        Retrieve the dataset from Hugging Face in a paginated manner until all rows are fetched.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns: ['sentence', 'tokens', 'labels'].

        Raises
        ------
        ConnectionError
            If there is a network-related error while making requests.
        Exception
            If the API responds with a non-200 status code.
        ValueError
            If the response does not contain the key 'rows'.
        """
        offset = 0
        accumulated_rows = []

        while True:
            params = {
                'dataset': self.dataset_name,
                'config': self.config,
                'split': self.split,
                'offset': offset,
                'length': self.length
            }
            logging.info(f"Requesting Hugging Face (offset={offset}, length={self.length})...")

            try:
                response = requests.get(
                    url=self.base_url,
                    params=params,
                    headers={'Authorization': f'Bearer {self.API_KEY}'}
                )
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"Error accessing Hugging Face: {e}")

            if response.status_code != 200:
                logging.error(f"Error response: {response.text}")
                raise Exception(
                    f"status_code={response.status_code}, reason={response.reason}"
                )

            json_data = response.json()
            if 'rows' not in json_data:
                raise ValueError("Response from Hugging Face does not contain 'rows'.")

            data = json_data['rows']
            # If no more data, break the loop
            if not data:
                break

            for item in data:
                if 'row' not in item:
                    logging.warning("Missing 'row' key in response item.")
                    continue
                row_data = item['row']

                # Ensure the required columns exist
                if not all(col in row_data for col in ('sentence', 'tokens', 'labels')):
                    logging.warning("Missing keys ('sentence', 'tokens', 'labels') in row data.")
                    continue

                accumulated_rows.append(row_data)

            logging.info(f"Loaded {len(data)} rows from this batch. Total so far: {len(accumulated_rows)}.")
            offset += self.length

        df_all = pd.DataFrame(accumulated_rows)
        logging.info(f"Total dimension of the dataset from Hugging Face: {df_all.shape}")
        return df_all

    def _preprocess_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate rows, ensuring consistent structure in tokens and labels.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns ['sentence', 'tokens', 'labels'].

        Returns
        -------
        pd.DataFrame
            A validated/corrected DataFrame with consistent tokens-labels lengths.
        """
        df_cleaned = pd.DataFrame(columns=['sentence', 'tokens', 'labels'])

        for index, row in df.iterrows():
            try:
                sentence = row['sentence']
                tokens = self._clean_list_string(row['tokens'])
                labels = self._clean_list_string(row['labels'])

                # Attempt to convert all labels to integers
                try:
                    labels = [int(x) for x in labels]
                except ValueError:
                    logging.warning(f"Not all labels are convertible to int (row {index}): {labels}")
                    continue

                # Ensure matching length of tokens and labels
                if len(tokens) != len(labels):
                    logging.warning(
                        f"Skipping row {index}: mismatch in length "
                        f"(tokens={len(tokens)}, labels={len(labels)})."
                    )
                    continue

                df_cleaned.loc[len(df_cleaned)] = [sentence, tokens, labels]

            except Exception as e:
                logging.warning(f"Unexpected error in row {index}: {e}")

        return df_cleaned

    def convert_df_from_huggingface_to_IOB(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the Hugging Face DataFrame from numeric labels to IOB format.

        Label definitions:
            - 0 -> 'O'
            - 1 -> 'B_mount'
            - 2 -> 'I_mount'

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns ['sentence', 'tokens', 'labels'].

        Returns
        -------
        pd.DataFrame
            A DataFrame with the same columns but 'labels' in IOB format.
        """
        cleaned_df = self._preprocess_labels(df)
        iob_data = []

        for _, row in cleaned_df.iterrows():
            sentence = row['sentence']
            tokens = row['tokens']
            labels = row['labels']

            iob_labels = []
            for label in labels:
                if label == 0:
                    iob_labels.append('O')
                elif label == 1:
                    iob_labels.append('B_mount')
                elif label == 2:
                    iob_labels.append('I_mount')
                else:
                    raise ValueError(f"Unknown label found: {label}")

            iob_data.append({
                'sentence': sentence,
                'tokens': tokens,
                'labels': iob_labels
            })

        return pd.DataFrame(iob_data)

    def convert_df_from_first_source_to_IOB(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a local dataset into IOB format. The local dataset is assumed to have:
            - text (the sentence)
            - marker (a list of tuples marking entity spans, e.g. [(start_idx, end_idx), ...])

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns ['text', 'marker'].

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['sentence', 'tokens', 'labels'] in IOB format.
        """

        def iob_labels_for_sentence(sentence: str, marker: str) -> Tuple[List[str], List[str]]:
            """
            Convert a sentence and marker into a list of tokens and IOB tags.

            Parameters
            ----------
            sentence : str
                The original text to split into tokens.
            marker : str
                String representation of a list of (start, end) tuples.

            Returns
            -------
            Tuple[List[str], List[str]]
                tokens, iob_tags
            """
            # Simple tokenization by words and punctuation
            words = re.findall(r"\w+|[.,!?;]", sentence)
            iob_tags = ["O"] * len(words)

            # Attempt to parse the marker as a list of tuples
            try:
                markers = ast.literal_eval(marker)
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Failed to parse marker: {marker}; error: {e}")
                return words, iob_tags

            # Assign B- or I- tags based on marker spans
            char_index = 0
            for i, word in enumerate(words):
                word_start = char_index
                word_end = char_index + len(word)
                for (start, end) in markers:
                    # If the token is fully within the entity span
                    if word_start >= start and word_end <= end:
                        if iob_tags[i] == 'O':
                            iob_tags[i] = 'B_mount'
                        else:
                            iob_tags[i] = 'I_mount'
                char_index = word_end + 1

            return words, iob_tags

        result_rows = []
        for _, row in df.iterrows():
            text = row.get('text', "")
            marker = row.get('marker', "[]")
            tokens, labels = iob_labels_for_sentence(text, marker)
            result_rows.append({
                'sentence': text,
                'tokens': tokens,
                'labels': labels
            })

        return pd.DataFrame(result_rows)

    def generate_dataset(self, df_local: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a final dataset by:
        1. Retrieving data from Hugging Face, converting it to IOB.
        2. Converting the local dataset to IOB.
        3. Concatenating both into a single DataFrame.

        Parameters
        ----------
        df_local : pd.DataFrame
            The local dataset with text and marker columns.

        Returns
        -------
        pd.DataFrame
            A concatenated DataFrame in IOB format.
        """
        logging.info("Start loading and processing the dataset from Hugging Face...")
        df_hf = self.get_data_from_huggingface()
        df_hf_iob = self.convert_df_from_huggingface_to_IOB(df_hf)
        logging.info("The Hugging Face dataset has been converted to IOB format.")

        logging.info("Starting processing of the local dataset...")
        df_local_iob = self.convert_df_from_first_source_to_IOB(df_local)
        logging.info("Local dataset converted to IOB format.")

        df_combined = pd.concat([df_hf_iob, df_local_iob], ignore_index=True)
        df_combined.reset_index(drop=True, inplace=True)
        logging.info(f"Final dataset size: {df_combined.shape[0]} rows.")
        return df_combined


def main() -> None:
    """
    Main function to orchestrate the dataset preparation.
    """
    configure_logging()

    # Instantiate the data preparation class
    data_preparation = DataPreparation(
        data_path_to_first_source='data/mountain_dataset_with_markup.csv',
        base_url='https://datasets-server.huggingface.co/rows'
    )

    # Read local data
    df_local = pd.read_csv(data_preparation.data_path_to_first_source)

    # Generate the combined IOB dataset
    df_IOB = data_preparation.generate_dataset(df_local)

    # Save to CSV
    output_path = 'data/general_mountains_ner_dataset_IOB.csv'
    df_IOB.to_csv(output_path, index=False)
    logging.info(f"Output saved at: {output_path}")


if __name__ == "__main__":
    main()
