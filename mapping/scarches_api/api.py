from flask import Flask, request, current_app
import os
import init as scarches
from threading import Thread
from utils import utils, parameters
import traceback
import subprocess

app = Flask(__name__)


def get_from_config(configuration, key):
    if key in configuration:
        return configuration[key]
    return None

import argparse
import os
import init as scarches
from threading import Thread
from utils import parameters
import ast

def get_from_config(configuration, key):
    if key in configuration:
        return configuration[key]
    return None

def query(config):
    run_async = get_from_config(config, parameters.RUN_ASYNCHRONOUSLY)
    if run_async is not None and run_async:
        actual_config = scarches.merge_configs(config)
        thread = Thread(target=scarches.query, args=(config,))
        thread.start()
        return actual_config, 200
    else:
        actual_configuration = scarches.query(config)
        return actual_configuration, 200

#execute cloud run job with query input

# @app.route("/query", methods=["POST"])
# def query():
#     """
#     Execute the desired Cloud Run Job with updated query.
#     """
#     try:

#         print("getting config!!!")
#         config = request.get_json(force=True)
#         actual_config = scarches.merge_configs(config)

#         print("got config!!!")
#         job_name = "archmap-data-1"
#         script_path = "query.py"
        
#         current_app.logger.info(
#             f"Updating the Cloud Run Job {job_name}"
#         )

#         # result = subprocess.run(
#         #     ["gcloud", "beta", "run", "jobs", "update",
#         #      job_name,
#         #      f" --command=['python', {script_path}, '--query {actual_config}']",
#         #     #  f" --args={actual_config}"
#         #     ],
#         #     capture_output=True,
#         #     text=True,
#         # )
#         # current_app.logger.info(
#         #   f"Std. Out: {result.stdout}\nStd. Error: {result.stderr}"
#         # )

#         # Triggering the job to actually run
#         current_app.logger.info(
#             f"Executing the Cloud Run Job {job_name}"
#         )
#         result = subprocess.run(
#             ["gcloud", "beta", "run", "jobs", "execute",
#              job_name, 
#              f" --command=['python', {script_path}, '--query {actual_config}']",
#              ],
#             capture_output=True,
#           text=True,
#         )

#         current_app.logger.info(
#           f"Std. Out: {result.stdout}\nStd. Error: {result.stderr}"
#         )

#         return "Cloud Run Job successfully triggered", 201
#     except Exception as e:
#         return f"Server Error: {e}", 500

     

# TODO: Notify that job is finished


@app.route("/liveness")
def liveness():
    return "up"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

    queryinfo = ast.literal_eval(os.environ["QUERY"])
    query(queryinfo)


