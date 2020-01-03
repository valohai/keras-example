import ast

import keras.layers

def add_layers(model, layers_list):
    tree = ast.parse(layers_list, 'asd', 'eval')

    for el in tree.body.elts:
        layer_name = el.func.id
        args = [ast.literal_eval(x) for x in el.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in el.keywords}
        try:
            layer_class = getattr(keras.layers, layer_name)
        except AttributeError as ex:
            raise RuntimeError('keras.layers.%s not found' % layer_name) from ex
        model.add(layer_class(*args, **kwargs))

    return model