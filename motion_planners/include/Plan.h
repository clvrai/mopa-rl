#ifndef _Planner_h
#define _Planner_h

#include <vector>
#include <string>
#include <iostream>


namespace MotionPlanner
{
    class Planner
    {
        public:
            std::string xml_filename;
            std::string algo;
            int num_actions;
            double sst_selection_radius;
            double sst_pruning_radius;
            Planner(std::string xml_filename, std::string algo, int num_actions, double sst_selection_radius, double sst_pruning_radius);
            ~Planner();
            std::vector<std::vector<double> > planning(std::vector<double> start_vec, std::vector<double> goal_vec, double timelimit);
            std::vector<std::vector<double> > planning_control(std::vector<double> start_vec, std::vector<double> goal_vec, double timelimit);
            std::vector<std::vector<double> > kinematic_planning(std::vector<double> start_vec, std::vector<double> goal_vec, double timelimit, double range);

    };
}

#endif
