import pandas as pd
from langchain_core.documents import Document

# This script converts the SHL assessment data into a format suitable for vector storage.

def dataconverter():
    assesment_data = pd.read_csv("../data/shl_assesments.csv")
    data = assesment_data[["Title", "Skills Assessed"]]

    assessment_list = []

    # Iterate over the rows of the DataFrame
    for index, row in data.iterrows():
        obj = {
            "assessment_title": row["Title"],
            "review": f"This assessment is designed to evaluate skills in {row['Skills Assessed']}."
        }
        assessment_list.append(obj)

    docs = []

    for obj in assessment_list:
        metadata = {"assessment_title": obj["assessment_title"]}
        page_content = obj["review"]

        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)

    return docs
