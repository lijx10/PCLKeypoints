#ifndef EXAMPLES_UTILS_H_
#define EXAMPLES_UTILS_H_

#include <fstream>
#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#include <Eigen/Dense>


class Point3f{
public:
    Point3f(float x, float y, float z) : x(x), y(y), z(z){
    }

    float x, y, z;
};



template <typename PointT, typename ContainerT>
void readPointsDat(const std::string& filename, ContainerT& points)
{
    std::ifstream in(filename.c_str());
    std::string line;
    boost::char_separator<char> sep(" ");
    // read point cloud from "freiburg format"
    while (!in.eof())
    {
        std::getline(in, line);
        in.peek();

        boost::tokenizer<boost::char_separator<char> > tokenizer(line, sep);
        std::vector<std::string> tokens(tokenizer.begin(), tokenizer.end());

        if (tokens.size() != 6) continue;
        float x = boost::lexical_cast<float>(tokens[3]);
        float y = boost::lexical_cast<float>(tokens[4]);
        float z = boost::lexical_cast<float>(tokens[5]);

        points.push_back(PointT(x, y, z));
    }

    in.close();
}

template <typename PointT, typename ContainerT>
void readPointsTxt(const std::string& filename, ContainerT& points)
{
    std::ifstream in(filename.c_str());
    std::string line;
    boost::char_separator<char> sep(" ");
    // read point cloud from "freiburg format"
    while (!in.eof())
    {
        std::getline(in, line);
        in.peek();

        boost::tokenizer<boost::char_separator<char> > tokenizer(line, sep);
        std::vector<std::string> tokens(tokenizer.begin(), tokenizer.end());

        if (tokens.size() != 14) continue;
        float x = boost::lexical_cast<float>(tokens[0]);
        float y = boost::lexical_cast<float>(tokens[1]);
        float z = boost::lexical_cast<float>(tokens[2]);

        points.push_back(PointT(x, y, z));
    }

    in.close();
}

template <typename PointT, typename ContainerT>
void readPoints(const std::string& filename, ContainerT& points)
{
    std::string surfix = filename.substr(filename.size()-3, filename.size());
    if(std::strcmp(surfix.c_str(), "dat")==0){
        readPointsDat<PointT>(filename, points);
        return;
    }
    if(std::strcmp(surfix.c_str(), "txt")==0) {
        readPointsTxt<PointT>(filename, points);
        return;
    }

    std::cout<<"wrong file!!!!!!!!!!!!!!!!!!"<<std::endl;
}


void printPoints(const Eigen::MatrixXf points){
    std::cout<<"Point number: "<<points.rows()<<std::endl;
    for(size_t i=0;i<points.rows();++i){
        std::cout<<points(i, 0)<<", "<<points(i, 1)<<", "<<points(i, 2)<<std::endl;
    }

}

#endif /* EXAMPLES_UTILS_H_ */
