#!/usr/bin/env python
import argparse

from flask import Flask, jsonify, request
from onmt.translate import OLServer, OLServerModelError
from sys import stderr

STATUS_OK = "ok"
STATUS_ERROR = "error"


def start(config_file,
          url_root="./translator",
          host="0.0.0.0",
          port=5000,
          debug=True):
    def prefix_route(route_function, prefix='', mask='{0}{1}'):
        def newroute(route, *args, **kwargs):
            return route_function(mask.format(prefix, route), *args, **kwargs)
        return newroute

    app = Flask(__name__)
    app.route = prefix_route(app.route, url_root)
    translation_server = OLServer()
    translation_server.start(config_file)

    @app.route('/models', methods=['GET'])
    def get_models():
        out = translation_server.list_models()
        stderr.write(str(out) + '\n')
        return jsonify(out)

    @app.route('/clone_model/<int:model_id>', methods=['POST'])
    def clone_model(model_id):
        out = {}
        data = request.get_json(force=True)
        timeout = -1
        if 'timeout' in data:
            timeout = data['timeout']
            del data['timeout']

        opt = data.get('opt', None)
        try:
            model_id, load_time = translation_server.clone_model(
                model_id, opt, timeout)
        except OLServerModelError as e:
            out['status'] = STATUS_ERROR
            out['error'] = str(e)
        else:
            out['status'] = STATUS_OK
            out['model_id'] = model_id
            out['load_time'] = load_time

        stderr.write(str(out) + '\n')
        return jsonify(out)

    @app.route('/unload_model/<int:model_id>', methods=['GET'])
    def unload_model(model_id):
        out = {"model_id": model_id}

        try:
            translation_server.unload_model(model_id)
            out['status'] = STATUS_OK
        except Exception as e:
            out['status'] = STATUS_ERROR
            out['error'] = str(e)

        stderr.write(str(out) + '\n')
        return jsonify(out)

    @app.route('/translate', methods=['POST'])
    def translate():
        inputs = request.get_json(force=True)
        out = {}
        try:
            translation, scores, n_best, times = translation_server.run(inputs)
            assert len(translation) == len(inputs)
            assert len(scores) == len(inputs)

            out = [[{"src": inputs[i]['src'], "tgt": translation[i],
                     "n_best": n_best,
                     "pred_score": scores[i]}
                    for i in range(len(translation))]]
        except OLServerModelError as e:
            out['error'] = str(e)
            out['status'] = STATUS_ERROR

        stderr.write(str(out) + '\n')
        return jsonify(out)

    @app.route('/train', methods=['POST'])
    def train():
        inputs = request.get_json(force=True)
        out = {}
        try:
            times = translation_server.train(inputs)

            out = [[{"src": inputs[i]['src'], "tgt": inputs[i]['tgt']}
                    for i in range(len(inputs))]]
        except OLServerModelError as e:
            out['error'] = str(e)
            out['status'] = STATUS_ERROR

        stderr.write(str(out) + '\n')
        return jsonify(out)

    @app.route('/to_cpu/<int:model_id>', methods=['GET'])
    def to_cpu(model_id):
        out = {'model_id': model_id}
        translation_server.models[model_id].to_cpu()

        out['status'] = STATUS_OK
        stderr.write(str(out) + '\n')
        return jsonify(out)

    @app.route('/to_gpu/<int:model_id>', methods=['GET'])
    def to_gpu(model_id):
        out = {'model_id': model_id}
        translation_server.models[model_id].to_gpu()

        out['status'] = STATUS_OK
        stderr.write(str(out) + '\n')
        return jsonify(out)

    app.run(debug=debug, host=host, port=port, use_reloader=False,
            threaded=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OpenNMT-py REST Server")
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="5000")
    parser.add_argument("--url_root", type=str, default="/translator")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--config", "-c", type=str,
                        default="./available_models/conf.json")
    args = parser.parse_args()
    start(args.config, url_root=args.url_root, host=args.ip, port=args.port,
          debug=args.debug)
