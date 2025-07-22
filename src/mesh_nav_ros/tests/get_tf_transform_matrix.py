#!/usr/bin/env python

import rospy
import tf
import sys
import numpy as np

def main():
    if len(sys.argv) != 3:
        print("Usage: python script_name.py parent_link child_link")
        return

    parent_link = sys.argv[1]
    child_link = sys.argv[2]

    rospy.init_node('tf_transform_node', anonymous=True)
    listener = tf.TransformListener()

    try:
        rospy.sleep(1.0)  # Allow time for the listener to initialize
        (trans, rot) = listener.lookupTransform(parent_link, child_link, rospy.Time(0))

        # Construct 4x4 transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = tf.transformations.quaternion_matrix(rot)[:3, :3]
        transformation[:3, 3] = np.array(trans)

        # Format the output
        matrix_as_string = ", ".join(
            [str(row.tolist()) for row in transformation]
        )
        print(f"[{matrix_as_string}]")

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
