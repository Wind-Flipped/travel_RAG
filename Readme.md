# Readme

## Setup
1. Create a conda environment and install packages in `requirements.txt`.
2. Get your api keys for OpenAI, Deepseek and set them as environment variables. Or use your local models.


### rag

If reasoning is based on the existing rag library, change the `llms.py`: `rag_database` parameter to the corresponding library

```bash
python llms.py 
```

If you need to create or update a rag library, go to `databases.py`: change the name of `cur_databases` to the new library name
```bash
python databases.py 
```

###Run
Run the main file
```bash
cd ./agents
python our_agent.py
```

