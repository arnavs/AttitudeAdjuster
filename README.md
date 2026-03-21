# LOU Attitude Adjuster

This is my submission for CMU's AI Poker Contest. It's named after the [Culture Warship](https://theculture.fandom.com/wiki/Attitude_Adjuster), and also because of what it does: blend a Deep CFR approach with a Bayesian posterior over opponent's holes (i.e., an attitude adjuster).




## How to run the engine

1. Create a virtual environment:

   ```bash
   python3.12 -m venv .venv
   ```

2. Activate the virtual environment:
   - On Windows:

     ```bash
     .venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source .venv/bin/activate
     ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

1. Basic coverage test:

```bash
pytest --cov=gym_env --cov-report=term-missing --cov-report=html --cov-branch
```

### Testing

1. To test the Attitude Adjuster against ProbabilityAgent, AllInAgent, FoldAgent, CallingStationAgent, RandomAgent:

```bash
python agent_test.py
```

2. To run a full match (1000 hands) of your agent against a specific agent:

```bash
python run.py
```

You can modify which bots play by modifying the agent config file. Write the file path to the corresponding agent for that bot to play. 
