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

### Generate Test Requests

**Note**: There is no `generate.py` file in this repository. The request generation functionality is provided by `generate_request.py` files.

#### Purpose of generate_request.py

The repository contains two `generate_request.py` files for generating synthetic travel requests:

1. **`rag/component/generate_request.py`** - Generates test requests for travel route planning
2. **`rag/databases/generate_request.py`** - Similar functionality for database-specific request generation

These scripts are used to:
- Extract existing travel routes and keywords from document data
- Randomly select target places from routes
- Calculate time and distance costs using the Poi2Route tool
- Generate human-like travel requests using AI (ZhipuAI GLM-4-air model)
- Output structured JSON files containing:
  - Original request text
  - AI-rewritten request (more natural language)
  - Route information
  - Target places
  - Time and distance constraints

**Example Output**: The scripts generate requests like:
```
请帮我推荐一条西湖一日游的旅游路线，要求是：[关键词]。除此，我还希望去[地点1]，[地点2]游玩。我希望我的交通时间在X分钟左右，交通距离在Y千米左右。
```

**Usage**: These generated requests are used for testing and evaluating the travel recommendation system's RAG capabilities.

### Run
Run the main file

```bash
cd ./agents
python our_agent.py
```

