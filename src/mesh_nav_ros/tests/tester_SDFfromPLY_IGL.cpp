#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <Eigen/Dense>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "point_cloud_and_closest_point_viz");
    ros::NodeHandle nh;
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);

    // Dummy example data: replace with your actual V, P, C
    Eigen::MatrixXd V(5, 3);
    V << 0,0,0,
         1,0,0,
         1,1,0,
         0,1,0,
         0.5,0.5,0;
    Eigen::Vector3d P(0.25, 0.25, 0);
    Eigen::Vector3d C(0.6, 0.6, 0);

    ros::Rate r(1);
    while (ros::ok())
    {
        // 1. Publish vertices as points (black)
        visualization_msgs::Marker points;
        points.header.frame_id = "map"; // or your fixed frame
        points.header.stamp = ros::Time::now();
        points.ns = "vertices";
        points.id = 0;
        points.type = visualization_msgs::Marker::POINTS;
        points.action = visualization_msgs::Marker::ADD;
        points.scale.x = 0.02; // size of points
        points.scale.y = 0.02;
        points.color.r = 0.0f;
        points.color.g = 0.0f;
        points.color.b = 0.0f;
        points.color.a = 1.0;

        for (int i = 0; i < V.rows(); ++i)
        {
            geometry_msgs::Point p;
            p.x = V(i,0);
            p.y = V(i,1);
            p.z = V(i,2);
            points.points.push_back(p);
        }

        // 2. Publish P as green sphere
        visualization_msgs::Marker sphere_P;
        sphere_P.header.frame_id = "map";
        sphere_P.header.stamp = ros::Time::now();
        sphere_P.ns = "point_P";
        sphere_P.id = 1;
        sphere_P.type = visualization_msgs::Marker::SPHERE;
        sphere_P.action = visualization_msgs::Marker::ADD;
        sphere_P.pose.position.x = P.x();
        sphere_P.pose.position.y = P.y();
        sphere_P.pose.position.z = P.z();
        sphere_P.scale.x = 0.05; // size of sphere
        sphere_P.scale.y = 0.05;
        sphere_P.scale.z = 0.05;
        sphere_P.color.r = 0.0f;
        sphere_P.color.g = 1.0f;
        sphere_P.color.b = 0.0f;
        sphere_P.color.a = 1.0;

        // 3. Publish C as red sphere
        visualization_msgs::Marker sphere_C;
        sphere_C.header.frame_id = "map";
        sphere_C.header.stamp = ros::Time::now();
        sphere_C.ns = "point_C";
        sphere_C.id = 2;
        sphere_C.type = visualization_msgs::Marker::SPHERE;
        sphere_C.action = visualization_msgs::Marker::ADD;
        sphere_C.pose.position.x = C.x();
        sphere_C.pose.position.y = C.y();
        sphere_C.pose.position.z = C.z();
        sphere_C.scale.x = 0.05;
        sphere_C.scale.y = 0.05;
        sphere_C.scale.z = 0.05;
        sphere_C.color.r = 1.0f;
        sphere_C.color.g = 0.0f;
        sphere_C.color.b = 0.0f;
        sphere_C.color.a = 1.0;

        // 4. Publish line between P and C (red)
        visualization_msgs::Marker line;
        line.header.frame_id = "map";
        line.header.stamp = ros::Time::now();
        line.ns = "line_PC";
        line.id = 3;
        line.type = visualization_msgs::Marker::LINE_STRIP;
        line.action = visualization_msgs::Marker::ADD;
        line.scale.x = 0.01; // line width
        line.color.r = 1.0f;
        line.color.g = 0.0f;
        line.color.b = 0.0f;
        line.color.a = 1.0;

        geometry_msgs::Point p_start, p_end;
        p_start.x = P.x();
        p_start.y = P.y();
        p_start.z = P.z();
        p_end.x = C.x();
        p_end.y = C.y();
        p_end.z = C.z();

        line.points.push_back(p_start);
        line.points.push_back(p_end);

        // Publish all
        marker_pub.publish(points);
        marker_pub.publish(sphere_P);
        marker_pub.publish(sphere_C);
        marker_pub.publish(line);

        ros::spinOnce();
        r.sleep();
    }

    return 0;
}