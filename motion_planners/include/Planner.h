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
            Planner(std::string xml_filename);
            ~Planner();
            std::vector<std::vector<double>> planning(std::vector<double> start_vec, std::vector<double> goal_vec, double timelimit);

    };
}

#endif
