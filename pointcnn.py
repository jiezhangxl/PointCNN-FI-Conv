from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pointfly as pf
import tensorflow as tf


def ficonv(pts, fts, qrs, tag, N, K1, mm, sigma, scale, K, D, P, C, C_pts_fts, kernel_num, is_training, with_kernel_registering, with_kernel_shape_comparison, 
          with_point_transformation, with_feature_transformation, with_learning_feature_transformation, kenel_initialization_method, depth_multiplier, sorting_method=None, with_global=False):
    Dis, indices_dilated = pf.knn_indices_general(qrs, pts, K*D, True)
    indices = indices_dilated[:, :, ::D, :]
  
    if sorting_method is not None:
        indices = pf.sort_points(pts, indices, sorting_method)

    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag+'nn_pts_local')  # (N, P, K, 3)
    if with_point_transformation or with_feature_transformation:
        X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K))
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
        X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None)
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
    if  with_point_transformation:
        if with_learning_feature_transformation:
            nn_pts_local = tf.matmul(X_2_KK, nn_pts_local)
            # Prepare features to be transformed
            nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
            nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)
        else:
            # Prepare features to be transformed
            nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
            nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)
            nn_pts_local = tf.matmul(X_2_KK, nn_pts_local)
    else:
        if with_learning_feature_transformation:
            nn_pts_local_ = tf.matmul(X_2_KK, nn_pts_local, name=tag+'nn_pts_local_')
            # Prepare features to be transformed
            nn_fts_from_pts_0 = pf.dense(nn_pts_local_, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
            nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)
        else:
            nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
            nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)

    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    P1 = tf.shape(nn_pts_local)[1]
    dim1 = 3
    if with_kernel_registering:
        ######################## preparing #########################
        if with_feature_transformation:
            nn_fts_input = tf.matmul(X_2_KK, nn_fts_input)

        r_data = tf.reduce_sum(nn_pts_local * nn_pts_local, axis=3, keep_dims=True, name=tag+'kernel_pow')
        ######################## kernel-registering #########################
        shape_id = 0
        if kenel_initialization_method == 'random':
            kernel_shape=tf.Variable(tf.random_uniform([K1,dim1], minval=-0.5, maxval=0.5, dtype=tf.float32), name=tag+'kernel_shape'+str(shape_id))
        else:
            kernel_shape=tf.Variable(tf.random_normal([K1,dim1], mean=0.0, stddev=1.0, dtype=tf.float32), name=tag+'kernel_shape'+str(shape_id))
        kernel_shape_dis = tf.sqrt(tf.reduce_sum(kernel_shape * kernel_shape, axis=1), name=tag+'kernel_shape_dis'+str(shape_id))
        kernel_shape_normal = scale * tf.div(kernel_shape,tf.reduce_max(kernel_shape_dis), name=tag+'kernel_shape_normal'+str(shape_id))

        r_kernel = tf.reduce_sum(kernel_shape_normal * kernel_shape_normal, axis=1, keep_dims=True, name=tag+'kernel_pow'+str(shape_id))
        reshape_data = tf.reshape(nn_pts_local, [N*P1*K,dim1], name=tag+'reshape_kernel'+str(shape_id))
        m = tf.reshape( tf.matmul(reshape_data, tf.transpose(kernel_shape_normal)), [N, P1, K, K1], name=tag+'mm'+str(shape_id))
        dis_matrix = tf.transpose(r_data-2*m+tf.transpose(r_kernel),perm=[0,1,3,2],name=tag+'dis_matrix'+str(shape_id))
        coef_matrix = tf.exp(tf.div(-dis_matrix,sigma), name=tag+'coef_matrix'+str(shape_id))
        #coef_matrix = tf.transpose(r_data-2*m+tf.transpose(r_kernel),perm=[0,1,3,2],name=tag+'coef_matrix'+str(shape_id))
        if with_kernel_shape_comparison:
            coef_global = tf.reduce_sum(coef_matrix, axis=[2,3], keep_dims=True)/K
            coef_normal = coef_global * tf.div(coef_matrix,tf.reduce_sum(coef_matrix , axis = 3 , keep_dims=True), name=tag+'coef_normal'+str(shape_id))
        else:
            coef_normal = tf.div(coef_matrix,tf.reduce_sum(coef_matrix , axis = 3 , keep_dims=True), name=tag+'coef_normal'+str(shape_id))
        
        fts_X = tf.matmul(coef_normal, nn_fts_input, name=tag+'fts_X'+str(shape_id))
        ###################################################################
        fts_conv = pf.separable_conv2d(fts_X, math.ceil(mm*C/kernel_num), tag+'fts_conv'+str(shape_id), is_training, (1, K1), depth_multiplier=depth_multiplier)
        fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag+'fts_conv_3d'+str(shape_id))
        
        for shape_id in range(kernel_num - 1):
            shape_id = shape_id + 1
            if kenel_initialization_method == 'random':
                kernel_shape=tf.Variable(tf.random_uniform([K1,dim1], minval=-0.5, maxval=0.5, dtype=tf.float32), name=tag+'kernel_shape'+str(shape_id))
            else:
                kernel_shape=tf.Variable(tf.random_normal([K1,dim1], mean=0.0, stddev=1.0, dtype=tf.float32), name=tag+'kernel_shape'+str(shape_id))
            kernel_shape_dis = tf.sqrt(tf.reduce_sum(kernel_shape * kernel_shape, axis=1), name=tag+'kernel_shape_dis'+str(shape_id))
            kernel_shape_normal = scale * tf.div(kernel_shape,tf.reduce_max(kernel_shape_dis), name=tag+'kernel_shape_normal'+str(shape_id))

            r_kernel = tf.reduce_sum(kernel_shape_normal * kernel_shape_normal, axis=1, keep_dims=True, name=tag+'kernel_pow'+str(shape_id))
            reshape_data = tf.reshape(nn_pts_local, [N*P1*K,dim1], name=tag+'reshape_kernel'+str(shape_id))
            m = tf.reshape( tf.matmul(reshape_data, tf.transpose(kernel_shape_normal)), [N, P1, K, K1], name=tag+'mm'+str(shape_id))
            dis_matrix = tf.transpose(r_data-2*m+tf.transpose(r_kernel),perm=[0,1,3,2],name=tag+'dis_matrix'+str(shape_id))
            coef_matrix = tf.exp(tf.div(-dis_matrix,sigma), name=tag+'coef_matrix'+str(shape_id))
            #coef_matrix = tf.transpose(r_data-2*m+tf.transpose(r_kernel),perm=[0,1,3,2],name=tag+'coef_matrix'+str(shape_id))
            if with_kernel_shape_comparison:
                coef_global = tf.reduce_sum(coef_matrix, axis=[2,3], keep_dims=True)/K
                coef_normal = coef_global * tf.div(coef_matrix,tf.reduce_sum(coef_matrix , axis = 3 , keep_dims=True), name=tag+'coef_normal'+str(shape_id))
            else:
                coef_normal = tf.div(coef_matrix,tf.reduce_sum(coef_matrix , axis = 3 , keep_dims=True), name=tag+'coef_normal'+str(shape_id))
            
            fts_X = tf.matmul(coef_normal, nn_fts_input, name=tag+'fts_X'+str(shape_id))
            ###################################################################
            fts_conv = pf.separable_conv2d(fts_X, math.ceil(mm*C/kernel_num), tag+'fts_conv'+str(shape_id), is_training, (1, K1), depth_multiplier=depth_multiplier)
            fts_conv_3d = tf.concat([fts_conv_3d, tf.squeeze(fts_conv, axis=2)], axis = -1 , name=tag+'fts_conv_3d'+str(shape_id))
    else:
        fts_X = nn_fts_input
        fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K), depth_multiplier=depth_multiplier)
        fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')

    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global', is_training)
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d

def xdeconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier,
          sorting_method=None, with_global=False):
    _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
    indices = indices_dilated[:, :, ::D, :]

    if sorting_method is not None:
        indices = pf.sort_points(pts, indices, sorting_method)

    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

    # Prepare features to be transformed
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)
    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    if with_X_transformation:
        ######################## X-transformation #########################
        X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K))
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
        X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None)
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
        ###################################################################
    else:
        fts_X = nn_fts_input

    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K), depth_multiplier=depth_multiplier)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')

    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global', is_training)
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d


class PointCNN:
    def __init__(self, points, features, is_training, setting):
        xconv_params = setting.xconv_params
        fc_params = setting.fc_params
        with_X_transformation = setting.with_X_transformation
        with_kernel_registering = setting.with_kernel_registering
        with_kernel_shape_comparison = setting.with_kernel_shape_comparison
        with_point_transformation = setting.with_point_transformation
        with_feature_transformation = setting.with_feature_transformation
        with_learning_feature_transformation = setting.with_learning_feature_transformation
        kenel_initialization_method = setting.kenel_initialization_method
        sorting_method = setting.sorting_method
        N = tf.shape(points)[0]

        kernel_num = setting.kernel_num

        if setting.sampling == 'fps':
            from sampling import tf_sampling

        self.layer_pts = [points]
        if features is None:
            self.layer_fts = [features]
        else:
            features = tf.reshape(features, (N, -1, setting.data_dim - 3), name='features_reshape')
            C_fts = xconv_params[0]['C'] // 2
            features_hd = pf.dense(features, C_fts, 'features_hd', is_training)
            self.layer_fts = [features_hd]

        # self.Dis = []
        # self.nn_pts_local = []
        for layer_idx, layer_param in enumerate(xconv_params):
            tag = 'xconv_' + str(layer_idx + 1) + '_'
            K1 = layer_param['K1']
            mm = layer_param['mm']
            sigma = layer_param['sigma']
            scale = layer_param['scale']
            K = layer_param['K']
            D = layer_param['D']
            P = layer_param['P']
            C = layer_param['C']
            links = layer_param['links']
            if setting.sampling != 'random' and links:
                print('Error: flexible links are supported only when random sampling is used!')
                exit()

            # get k-nearest points
            pts = self.layer_pts[-1]
            fts = self.layer_fts[-1]
            if P == -1 or (layer_idx > 0 and P == xconv_params[layer_idx - 1]['P']):
                qrs = self.layer_pts[-1]
            else:
                if setting.sampling == 'fps':
                    fps_indices = tf_sampling.farthest_point_sample(P, pts)
                    batch_indices = tf.tile(tf.reshape(tf.range(N), (-1, 1, 1)), (1, P, 1))
                    indices = tf.concat([batch_indices, tf.expand_dims(fps_indices,-1)], axis=-1)
                    qrs = tf.gather_nd(pts, indices, name= tag + 'qrs') # (N, P, 3)
                elif setting.sampling == 'ids':
                    indices = pf.inverse_density_sampling(pts, K, P)
                    qrs = tf.gather_nd(pts, indices)
                elif setting.sampling == 'random':
                    qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
                else:
                    print('Unknown sampling method!')
                    exit()
            self.layer_pts.append(qrs)

            if layer_idx == 0:
                C_pts_fts = C // 2 if fts is None else C // 4
                depth_multiplier = 4
            else:
                C_prev = xconv_params[layer_idx - 1]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)
            with_global = (setting.with_global and layer_idx == len(xconv_params) - 1)
            fts_xconv= ficonv(pts, fts, qrs, tag, N, K1, mm, sigma, scale, K, D, P, C, C_pts_fts, kernel_num, is_training, with_kernel_registering, with_kernel_shape_comparison,
                              with_point_transformation, with_feature_transformation, with_learning_feature_transformation, kenel_initialization_method, depth_multiplier, sorting_method, with_global)
            #self.Dis.append(Dis_) 
            #self.nn_pts_local.append(nn_pts_local_)

            fts_list = []
            for link in links:
                fts_from_link = self.layer_fts[link]
                if fts_from_link is not None:
                    fts_slice = tf.slice(fts_from_link, (0, 0, 0), (-1, P, -1), name=tag + 'fts_slice_' + str(-link))
                    fts_list.append(fts_slice)
            if fts_list:
                fts_list.append(fts_xconv)
                self.layer_fts.append(tf.concat(fts_list, axis=-1, name=tag + 'fts_list_concat'))
            else:
                self.layer_fts.append(fts_xconv)

        if hasattr(setting, 'xdconv_params'):
            for layer_idx, layer_param in enumerate(setting.xdconv_params):
                tag = 'xdconv_' + str(layer_idx + 1) + '_'
                K = layer_param['K']
                D = layer_param['D']
                pts_layer_idx = layer_param['pts_layer_idx']
                qrs_layer_idx = layer_param['qrs_layer_idx']

                pts = self.layer_pts[pts_layer_idx + 1]
                fts = self.layer_fts[pts_layer_idx + 1] if layer_idx == 0 else self.layer_fts[-1]
                qrs = self.layer_pts[qrs_layer_idx + 1]
                fts_qrs = self.layer_fts[qrs_layer_idx + 1]
                P = xconv_params[qrs_layer_idx]['P']
                C = xconv_params[qrs_layer_idx]['C']
                C_prev = xconv_params[pts_layer_idx]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = 1
                fts_xdconv = xdeconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                                   depth_multiplier, sorting_method)
                fts_concat = tf.concat([fts_xdconv, fts_qrs], axis=-1, name=tag + 'fts_concat')
                fts_fuse = pf.dense(fts_concat, C, tag + 'fts_fuse', is_training)
                self.layer_pts.append(qrs)
                self.layer_fts.append(fts_fuse)

        self.fc_layers = [self.layer_fts[-1]]
        for layer_idx, layer_param in enumerate(fc_params):
            C = layer_param['C']
            dropout_rate = layer_param['dropout_rate']
            fc = pf.dense(self.fc_layers[-1], C, 'fc{:d}'.format(layer_idx), is_training)
            fc_drop = tf.layers.dropout(fc, dropout_rate, training=is_training, name='fc{:d}_drop'.format(layer_idx))
            self.fc_layers.append(fc_drop)
