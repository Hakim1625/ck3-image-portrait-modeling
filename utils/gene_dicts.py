import json

dicts = [{
		'age':0,
		'skincolor_light':0,
		'skincolor_dark':0,
 		'gene_chin_forward':0,
 		'gene_chin_height':0,
 		'gene_chin_width':0,
 		'gene_eye_angle':0,
 		'gene_eye_depth':0,
 		'gene_eye_height':0,
 		'gene_eye_distance':0,
 		'gene_eye_shut':0,
 		'gene_forehead_angle':0,
 		'gene_forehead_brow_height':0,
 		'gene_forehead_roundness':0,
 		'gene_forehead_width':0,
 		'gene_forehead_height':0,
 		'gene_head_height':0,
 		'gene_head_width':0,
 		'gene_head_profile':0,
 		'gene_head_top_height':0,
 		'gene_head_top_width':0,
 		'gene_jaw_angle':0,
 		'gene_jaw_forward':0,
 		'gene_jaw_height':0,
 		'gene_jaw_width':0,
 		'gene_mouth_corner_depth':0,
 		'gene_mouth_corner_height':0,
 		'gene_mouth_forward':0,
 		'gene_mouth_height':0,
 		'gene_mouth_width':0,
 		'gene_mouth_upper_lip_size':0,
 		'gene_mouth_lower_lip_size':0,
 		'gene_mouth_open':0,
 		'gene_neck_length':0,
 		'gene_neck_width':0,
 		'gene_bs_cheek_forward':0,
 		'gene_bs_cheek_height':0,
 		'gene_bs_cheek_width':0,
 		'gene_bs_ear_angle':0,
 		'gene_bs_ear_inner_shape':0,
 		'gene_bs_ear_bend':0,
 		'gene_bs_ear_outward':0,
 		'gene_bs_ear_size':0,
 		'gene_bs_eye_corner_depth':0,
 		'gene_bs_eye_fold_shape':0,
 		'gene_bs_eye_size':0,
 		'gene_bs_eye_upper_lid_size':0,
 		'gene_bs_forehead_brow_curve':0,
 		'gene_bs_forehead_brow_forward':0,
 		'gene_bs_forehead_brow_inner_height':0,
 		'gene_bs_forehead_brow_outer_height':0,
 		'gene_bs_forehead_brow_width':0,
 		'gene_bs_jaw_def':0,
 		'gene_bs_mouth_lower_lip_def':0,
 		'gene_bs_mouth_lower_lip_full':0,
 		'gene_bs_mouth_lower_lip_pad':0,
 		'gene_bs_mouth_lower_lip_width':0,
 		'gene_bs_mouth_philtrum_def':0,
 		'gene_bs_mouth_philtrum_shape':0,
 		'gene_bs_mouth_philtrum_width':0,
 		'gene_bs_mouth_upper_lip_def':0,
 		'gene_bs_mouth_upper_lip_full':0,
 		'gene_bs_mouth_upper_lip_profile':0,
 		'gene_bs_mouth_upper_lip_width':0,
 		'gene_bs_nose_forward':0,
 		'gene_bs_nose_height':0,
 		'gene_bs_nose_length':0,
 		'gene_bs_nose_nostril_height':0,
 		'gene_bs_nose_nostril_width':0,
 		'gene_bs_nose_profile':0,
 		'gene_bs_nose_ridge_angle':0,
 		'gene_bs_nose_ridge_width':0,
 		'gene_bs_nose_size':0,
 		'gene_bs_nose_tip_angle':0,
 		'gene_bs_nose_tip_forward':0,
 		'gene_bs_nose_tip_width':0,
 		'face_detail_cheek_def':0,
 		'face_detail_cheek_fat':0,
 		'face_detail_chin_cleft':0,
 		'face_detail_chin_def':0,
 		'face_detail_eye_lower_lid_def':0,
 		'face_detail_eye_socket':0,
 		'face_detail_nasolabial':0,
 		'face_detail_nose_ridge_def':0,
 		'face_detail_nose_tip_def':0,
 		'face_detail_temple_def':0,
 		'expression_brow_wrinkles':0,
 		'expression_eye_wrinkles':0,
 		'expression_forehead_wrinkles':0,
 		'expression_other':0,
 		'complexion':0,
 		'gene_height':0,
 		'gene_bs_body_type':0,
 		'gene_bs_body_shape':0,
 		'gene_age':0,
 		'gene_eyebrows_shape':0,
 		'gene_eyebrows_fullness':0,
 		'gene_body_hair':0,
		'gene_baldness':0,
		'eyelashes_accessory':0
},
 {'gene_age': ['old_1', 'old_2', 'old_3', 'old_4', 'old_beauty_1'],
'gene_bs_body_type': ["body_average", "body_fat_head_fat_low", "body_fat_head_fat_medium", "body_fat_head_fat_full"],
'gene_body_hair': ["body_hair_sparse", "body_hair_avg", "body_hair_dense", "body_hair_sparse_low_stubble", "body_hair_avg_low_stubble", "body_hair_dense_low_stubble"],
'complexion': ["complexion_1", "complexion_2", "complexion_3", "complexion_4", "complexion_5", "complexion_6", "complexion_7", "complexion_beauty_1", "complexion_ugly_1"],
'gene_bs_ear_bend': ["ear_lower_bend_pos", "ear_upper_bend_pos", "ear_both_bend_pos"],
'face_detail_cheek_def': ["cheek_def_01", "cheek_def_02"],
'face_detail_cheek_fat': ["cheek_fat_01_pos", "cheek_fat_02_pos", "cheek_fat_03_pos", "cheek_fat_04_pos",  "cheek_fat_01_neg"],
'face_detail_chin_cleft': ["chin_cleft",  "chin_dimple"],
'expression_brow_wrinkles': ["brow_wrinkles_01", "brow_wrinkles_02", "brow_wrinkles_03", "brow_wrinkles_04"],
'expression_eye_wrinkles': ["eye_wrinkles_01", "eye_wrinkles_02", "eye_wrinkles_03"],
'expression_forehead_wrinkles': ["forehead_wrinkles_01", "forehead_wrinkles_02", "forehead_wrinkles_03"],
'expression_other': ["cheek_wrinkles_left_01", "cheek_wrinkles_right_01", "cheek_wrinkles_both_01", "nose_wrinkles_01"],
'face_detail_eye_socket': ["eye_socket_01", "eye_socket_02", "eye_socket_03"],
'gene_eyebrows_fullness': ["no_eyebrows", "layer_2_avg_thickness", "layer_2_high_thickness", "layer_2_low_thickness", "layer_2_lower_thickness"],
'gene_eyebrows_shape': ["no_eyebrows",  "avg_spacing_avg_thickness", "avg_spacing_high_thickness", "avg_spacing_low_thickness", "avg_spacing_lower_thickness", "far_spacing_avg_thickness", "far_spacing_high_thickness", "far_spacing_low_thickness", "far_spacing_lower_thickness", "close_spacing_avg_thickness", "close_spacing_high_thickness", "close_spacing_low_thickness", "close_spacing_lower_thickness"],
'face_detail_nasolabial': ["nasolabial_01", "nasolabial_02", "nasolabial_03", "nasolabial_04"],
'gene_bs_nose_profile': ["nose_profile_neg", "nose_profile_pos", "nose_profile_hawk", "nose_profile_hawk_pos"],
'face_detail_nose_ridge_def': ["nose_ridge_def_pos", "nose_ridge_def_neg"],
'gene_baldness': ["no_baldness", "male_pattern_baldness"],
'eyelashes_accessory': ["no_eyelashes", "normal_eyelashes", "asian_eyelashes"]
}]

dicts[0].update(dicts[1])
dicts[0] = { key:0 for key in dicts[0].keys()}
print(len(dicts[0].keys()))
json = json.dumps(dicts)

f = open("gene_dicts.json","w")
f.write(json)
f.close()