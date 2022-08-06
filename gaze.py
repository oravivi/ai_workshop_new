import json
import workshop_utils

num_of_frames=2
print(workshop_utils.extract_coordinates_for_all_frames(0,2,["eyelid_top_right","eyelid_bottom_right",
                                                             "eyelid_top_left","eyelid_bottom_left",
                                                             "eye_right_right","eye_right_left",
                                                             "eye_left_right","eye_left_left",
                                                             "eye_right_pupil","eye_left_pupil",
                                                             "head_side_left","head_side_right"
                                                             "nose"],
                                                        [37,41,44,46,45,42,36,39,69,68,0,16,27]))

