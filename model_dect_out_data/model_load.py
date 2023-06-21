import os

from backend.support.support_func import normalize, sigmoid
import tensorflow as tf
import numpy as np


class BoneDetectionModel:

    def __init__(self):
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        self.model_composition = dict()
        self.model_composition["detect_human_part_list"] = ['nose', 'leye', 'reye', 'lear', 'rear', 'lshoulder',
         'rshoulder', 'lelbow', 'relbow', 'lwrist', 'rwrist', 'lhip', 'rhip', 'lknee',
         'rknee', 'lankle', 'rankle']


    def convert_pose_to_output(self,keypoints_person, img_height, img_width):
        # get heatmap and offsets
        heatmaps = keypoints_person[2].reshape((1, 23, 17, 17))
        heatmaps_shape = heatmaps.shape
        offset = keypoints_person[0].reshape((1, 23, 17, 34))

        # get number of boxes
        height = heatmaps_shape[1]
        width = heatmaps_shape[2]
        num_keypoints = heatmaps_shape[3]

        # get box pos of key points
        keypoint_positions = []
        for keypoint in range(num_keypoints):
            max_val = heatmaps[0][0][0][keypoint]
            max_row = 0
            max_col = 0
            for row in range(height):
                for col in range(width):
                    if heatmaps[0][row][col][keypoint] > max_val:
                        max_val = heatmaps[0][row][col][keypoint]
                        max_row = row
                        max_col = col

            keypoint_positions.append((max_row, max_col))

        args = np.zeros((17, 3))

        for i, position in enumerate(keypoint_positions):
            position_y = position[0]
            position_x = position[1]

            args[i][0] = int(position_y / float(height - 1.0) * img_height + offset[0][position_y][position_x][i])

            args[i][1] = int(
                position_x / float(width - 1.0) * img_width + offset[0][position_y][position_x][i + num_keypoints])

            args[i][2] = sigmoid(heatmaps[0][position_y][position_x][i])
        return args

    def model_init(self, parseargs):
        move_dict = dict()
        move_dict["thunder"] = 256
        move_dict["lightning"] = 192
        model_name = parseargs.model
        args = [parseargs.submodel, parseargs.istf_lite, parseargs.tf_quant, parseargs.load_type]
        if model_name == "movenet":
            model_type = args[0]
            is_lite = args[1]
            type_quant = args[2]
            load_type = args[3]

            size = move_dict[model_type]
            model_height = size
            model_width = size

            if is_lite:
                if load_type == "local":
                    interpreter = tf.lite.Interpreter(model_path="model.tflite")  # 加载模型
                    interpreter.allocate_tensors()  # 分配内存
                    module = tf.saved_model.load('./')

                    def model(input_image, img_height, img_width):
                        model = module.signatures['serving_default']
                        # SavedModel format expects tensor type of int32.
                        input_image = tf.cast(input_image, dtype=tf.int32)
                        # Run model inference.
                        outputs = model(input_image)
                        # Output is a [1, 1, 17, 3] tensor.
                        keypoints_with_scores = outputs['output_0'].numpy()
                        tmp_array = np.reshape(keypoints_with_scores, (17, 3))
                        arg = np.zeros((17, 3))
                        arg[:, 0] = tmp_array[:, 0] * img_height
                        arg[:, 1] = tmp_array[:, 1] * img_width
                        arg[:, 2] = tmp_array[:, 2]
                        return arg

        elif model_name == "posenet":
            import tensorrt as trt
            import common
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)
            trt.init_libnvinfer_plugins(TRT_LOGGER, "")
            def get_engine(engine_file_path):
                print("Reading engine from file {}".format(engine_file_path))
                with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                    return runtime.deserialize_cuda_engine(f.read())
            engine = get_engine("posenet_engine.trt")
            context = engine.create_execution_context()
            inputs, outputt, bindings, stream = common.allocate_buffers(engine)
            model_height = 353
            model_width = 257
            def model(input_image, img_height, img_width):
                normalized_im = np.expand_dims(input_image, axis=0)
                normalized_im = normalize(normalized_im)
                inputs[0].host = normalized_im
                outputs = common.do_inference_v2(
                    context, bindings=bindings, inputs=inputs, outputs=outputt, stream=stream
                )
                argus = self.convert_pose_to_output(outputs, img_height, img_width)
                return argus

        return model, model_height, model_width, self.model_composition
