"""
File containing any custom layers that are not included in keras

"""
from keras.layers import Lambda


def correlation_layer(convIn_left, convIn_right, shape, max_displacement=20, stride=2):
    # Implementation of correlation layer from Flownet
    widthIn, depthIn, heightIn = *shape
    layer_list = []
    dotLayer = dotLayer()
    for x_disp in range(-max_displacement, max_displacement+stride, stride):
        for y_disp in range(-max_displacement, max_displacement+stride, stride):
            for z_disp in range(-max_displacement, max_displacement+stride, stride):
                padded_stride = get_padded_stride(
                    convIn_right, x_disp, y_disp, z_disp, widthIn, depthIn, heightIn)
                current_layer = dotLayer([convIn_left, padded_stride])
                layer_list.append(current_layer)
    return Lambda(lambda x: tf.concat(x, axis=-1), name='output_correlation')(layer_list)


def dotLayer():
    # Check axis to reduce sum along
    # Dot product operation
    return Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0], x[1]), axis=-1, keep_dims=True), name='DotLayer')


def get_padded_stride(inputLayer, x_displacement, y_displacement, z_displacement, widthIn, depthIn, heightIn):
    slice_width = widthIn - abs(x_displacement)
    slice_height = heightIn - abs(y_displacement)
    slice_depth = depthIn - abs(z_displacement)
    start_x = abs(x_displacement) if x_displacement < 0 else 0
    start_y = abs(y_displacement) if y_displacement < 0 else 0
    start_z = abs(z_displacement) if z_displacement < 0 else 0
    left_pad = x_displacement if x_displacement > 0 else 0
    right_pad = start_x
    top_pad = z_displacement if z_displacement > 0 else 0
    bottom_pad = start_z
    front_pad = y_displacement if y_displacement > 0 else 0
    back_pad = start_y

    get_layer = Lambda(lambda x: tf.pad(x[:, start_x:slice_width, start_y:slice_depth, start_z:slice_height, :],
                        paddings=[[0, 0], [left_pad, right_pad], [
                            front_pad, back_pad], [top_pad, bottom_pad], [0, 0]],
                        name="gather_{}_{}_{}".format(x_displacement, y_displacement, z_displacement))(inputLayer)
    return get_layer
