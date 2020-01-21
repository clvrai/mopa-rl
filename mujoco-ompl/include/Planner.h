#ifndef _Planner_h
#define _Planner_h

class Planner
{
    public:
        string xml_filename;
        Planner(string xml_filename);
        ~Planner();
        int plan(vector<double> start_vec, vector<double> goal_vec, double timelimit);

};

#endif
