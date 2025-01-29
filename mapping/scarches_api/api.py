# from flask import Flask, request
import os
import init as scarches
from threading import Thread
from scarches_api.utils import utils, parameters
import traceback
import json

# app = Flask(__name__)


def get_from_config(configuration, key):
    if key in configuration:
        return configuration[key]
    return None

def main():
    try:
        config = os.environ
        print(config) 

        # convert listed vars
        vars_to_convert = config["vars_to_convert"]
        vars_to_convert = vars_to_convert.replace("'", '"')
        vars_to_convert = json.loads(vars_to_convert)
        for key, values in vars_to_convert.items():
            if key=="dictionary":
                for v in values:
                    value = config[v]
                    value = value.replace("'", '"')
                    config[v] = json.loads(value)

            if key=="integer":
                for v in values:
                    config[v] = int(v)
            
            if key=="boolean":
                for v in values:
                    config[v] = {"true": True, "false": False}.get(config[v].lower(), False)

        run_async = get_from_config(config, parameters.RUN_ASYNCHRONOUSLY)
        if run_async is not None and run_async:
            actual_config = scarches.merge_configs(config)
            # thread = Thread(target=scarches.query, args=(config,))
            # thread.start()
            actual_config = scarches.query(config)
            return actual_config, 200
        else:
            actual_configuration = scarches.query(config)
            return actual_configuration, 200
    except Exception as e:
        print("Error in query\n")
        traceback.print_exc()
        if e is not None:
            if len(str(e)) > 0:
                utils.notify_backend(get_from_config(config, parameters.WEBHOOK), {'error': str(e)})
            else:
                utils.notify_backend(get_from_config(config, parameters.WEBHOOK), {'error': "Unknown error"})
        else:
            utils.notify_backend(get_from_config(config, parameters.WEBHOOK), {'error': "Unknown error"})
        
        return {'error': str(e)}, 500



# @app.route('/query', methods=['POST'])
# def query():
#     try:
#         config = request.get_json(force=True)
#         run_async = get_from_config(config, parameters.RUN_ASYNCHRONOUSLY)
#         if run_async is not None and run_async:
#             actual_config = scarches.merge_configs(config)
#             # thread = Thread(target=scarches.query, args=(config,))
#             # thread.start()
#             actual_config = scarches.query(config)
#             return actual_config, 200
#         else:
#             actual_configuration = scarches.query(config)
#             return actual_configuration, 200
#     except Exception as e:
#         print("Error in query\n")
#         traceback.print_exc()
#         if e is not None:
#             if len(str(e)) > 0:
#                 utils.notify_backend(get_from_config(config, parameters.WEBHOOK), {'error': str(e)})
#             else:
#                 utils.notify_backend(get_from_config(config, parameters.WEBHOOK), {'error': "Unknown error"})
#         else:
#             utils.notify_backend(get_from_config(config, parameters.WEBHOOK), {'error': "Unknown error"})
        
#         return {'error': str(e)}, 500


# @app.route("/liveness")
# def liveness():
#     return "up"


if __name__ == "__main__":
    main()

    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
