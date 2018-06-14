#include <iostream>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeAdvect.h>
#include <openvdb/tools/PointAdvect.h>
#include <openvdb/tools/PointScatter.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/math/Transform.h>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/IterativeLinearSolvers>
#include <mutex>
#include <time.h>
#include "deformHeader.h"

clock_t end;
double cpu_time_used;
clock_t start = clock();
typedef Eigen::Triplet<double> Triplet;
double factor = 1.0;
double spline(double x)
{
    x -= 0.5;
    if (x < 0)
    {
        x *= -1.0*factor;
    }
    if (x < 0.5*factor)
    {
        return 1.0 * (4.0*x*x*x/(factor*factor*factor) - 4.0*x*x/(factor*factor) + 2.0/3.0);
    }
    if (x <= 1.0*factor)
    {
        return 1.0* ((-8.0 * (x * x * x)/(6.0*factor*factor*factor)) + 4.0*x*x/(factor*factor) - 4.0*x/(factor) + 4.0/3.0);
    }
    return 0;
}
bool isWithinBounds(openvdb::Coord xyz, int bound)
{
    if (fabs(xyz.x()) > bound || fabs(xyz.y()) > bound || fabs(xyz.z()) > bound)
    {
        return false;
    }
    return true;
}
bool isSolid(openvdb::Coord xyz, openvdb::FloatGrid::Ptr solidGrid)
{
    openvdb::FloatGrid::Accessor saccessor = solidGrid->getAccessor();
    if (saccessor.getValue(xyz) == 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}


openvdb::Vec3d getVelocity(openvdb::Coord c, openvdb::Vec3dGrid::Ptr velocity)
{
    openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
    openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
    openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);
    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::Vec3d vel = vaccessor.getValue(c);
    double u = (vel.x() + vaccessor.getValue(ipjk).x())/2.0;
    double v = (vel.y() + vaccessor.getValue(ijpk).y())/2.0;
    double w = (vel.z() + vaccessor.getValue(ijkp).z())/2.0;
    return openvdb::Vec3d(u, v, w);
}

openvdb::Vec3d clampedCatmullRom(openvdb::Vec3d c, openvdb::Vec3dGrid::Ptr velocity, int bound, openvdb::FloatGrid::Ptr fluidGrid)
{
    int fcx = round(c.x());
    int fcy = round(c.y());
    int fcz = round(c.z());

    int minx = fcx - 1 > -1 * bound ? fcx - 1 : -1 * bound;
    int miny = fcy - 1 > -1 * bound ? fcy - 1 : -1 * bound;
    int minz = fcz - 1 > -1 * bound ? fcz - 1 : -1 * bound;
    int maxx = fcx + 1 < bound ? fcx + 1 : bound;
    int maxy = fcy + 1 < bound ? fcy + 1 : bound;
    int maxz = fcz + 1 < bound ? fcz + 1 : bound;

    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::FloatGrid::Accessor faccessor = fluidGrid->getAccessor();

    double u = 0;
    double v = 0;
    double w = 0;

    double weight = 0;

    openvdb::Vec3d velc = vaccessor.getValue(openvdb::Coord(fcx, fcy, fcz));

    double minu = velc.x();
    double minv = velc.y();
    double minw = velc.z();
    double maxu = velc.x();
    double maxv = velc.y();
    double maxw = velc.z();

    for (int x = minx; x <= maxx; ++x)
    {
        for (int y = miny; y <= maxy; ++y)
        {
            for (int z = minz; z <= maxz; ++z)
            {
                if (isWithinBounds(openvdb::Coord(x, y, z), 13))
                {
                    velc = getVelocity(openvdb::Coord(x, y, z), velocity);

                    minu = velc.x() < minu ? velc.x() : minu;
                    minv = velc.y() < minv ? velc.y() : minv;
                    minw = velc.z() < minw ? velc.z() : minw;

                    maxu = velc.x() > maxu ? velc.x() : maxu;
                    maxv = velc.y() > maxv ? velc.y() : maxv;
                    maxw = velc.z() > maxw ? velc.z() : maxw;

                    double mass = faccessor.getValue(openvdb::Coord(x, y, z));
                    // if (mass > 0)
                    // {
                        double cw = spline(c.x() - x) * spline(c.y() - y) * spline(c.z() - z);//should it be mass*?
                        weight += cw;
                        u += velc.x() * cw;
                        v += velc.y() * cw;
                        w += velc.z() * cw;
                // }
                }
            }
        }
    }
    // weight = 1.0;//experiment
    if (weight != 0)
    {
        u /= weight;
        v /= weight;
        w /= weight;
    }
    else
    {
        return openvdb::Vec3d(0, 0, 0);
    }

    // u = u > maxu ? maxu : u;
    // v = v > maxv ? maxv : v;
    // w = w > maxw ? maxw : w;

    // u = u < minu ? minu : u;
    // v = v < minv ? minv : v;
    // w = w < minw ? minw : w;

    return openvdb::Vec3d(u, v, w);
}


openvdb::Vec3d CatmullRomFLIP(openvdb::Vec3d c, openvdb::Vec3d vel, openvdb::Vec3dGrid::Ptr velocity, int bound, openvdb::Vec3dGrid::Ptr velBeforeUpdate, openvdb::FloatGrid::Ptr fluidGrid, openvdb::FloatGrid::Ptr solidGrid)
{
    int fcx = round(c.x());
    int fcy = round(c.y());
    int fcz = round(c.z());

    int minx = fcx - 1 > -1 * bound ? fcx - 1 : -1 * bound;
    int miny = fcy - 1 > -1 * bound ? fcy - 1 : -1 * bound;
    int minz = fcz - 1 > -1 * bound ? fcz - 1 : -1 * bound;
    int maxx = fcx + 1 < bound ? fcx + 1 : bound;
    int maxy = fcy + 1 < bound ? fcy + 1 : bound;
    int maxz = fcz + 1 < bound ? fcz + 1 : bound;

    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::Vec3dGrid::Accessor vpaccessor = velBeforeUpdate->getAccessor();
    openvdb::FloatGrid::Accessor faccessor = fluidGrid->getAccessor();
    double weight = 0;
    openvdb::Vec3d velc;
    openvdb::Vec3d velp;
    openvdb::Vec3d delta(0, 0, 0);

    for (int x = minx; x <= maxx; ++x)
    {
        for (int y = miny; y <= maxy; ++y)
        {
            for (int z = minz; z <= maxz; ++z)
            {
                if(isWithinBounds(openvdb::Coord(x, y, z), 13))
                {
                    velc = getVelocity(openvdb::Coord(x, y, z), velocity);
                    velp = getVelocity(openvdb::Coord(x, y, z), velBeforeUpdate);

                    double mass = faccessor.getValue(openvdb::Coord(x, y, z));
                    // if (mass > 0.1)
                    // {
                        double cw = spline(c.x() - x) * spline(c.y() - y) * spline(c.z() - z);//should it be mass*?
                        // cw = mass;
                        weight += cw;
                        
                        // cw = 1.0;
                        // weight = 1.0;//experi
                        // The difference is not between current particle velocity and grid velocity, but it's between previous and current grid velocities
                        delta += (velc - velp) * cw;
                    // }
                }
            }
        }
    }
    if (weight == 0)
    {
        return openvdb::Vec3d(0, 0, 0);
    }
    return delta/weight;
}

void p2gCatmullRom(openvdb::Vec3d c, openvdb::Vec3d velc, openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr weights, int bound, std::mutex ***locks, openvdb::FloatGrid::Ptr solidGrid)
{
    int fcx = round(c.x());
    int fcy = round(c.y());
    int fcz = round(c.z());

    int minx = fcx - 1 > -1 * bound ? fcx - 1 : -1 * bound;
    int miny = fcy - 1 > -1 * bound ? fcy - 1 : -1 * bound;
    int minz = fcz - 1 > -1 * bound ? fcz - 1 : -1 * bound;
    int maxx = fcx + 1 < bound ? fcx + 1 : bound;
    int maxy = fcy + 1 < bound ? fcy + 1 : bound;
    int maxz = fcz + 1 < bound ? fcz + 1 : bound;

    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::FloatGrid::Accessor waccessor = weights->getAccessor();

    for (int x = minx; x <= maxx; ++x)
    {
        for (int y = miny; y <= maxy; ++y)
        {
            for (int z = minz; z <= maxz; ++z)
            {
                openvdb::Coord tc = openvdb::Coord(x, y, z);
                if (!isSolid(tc, solidGrid) && isWithinBounds(tc, bound - 2))
                {
                    locks[x+bound][y+bound][z+bound].lock();
                    double cw = spline(c.x() - x) * spline(c.y() - y) * spline(c.z() - z);
                    waccessor.setValue(tc, waccessor.getValue(tc) + cw);
                    vaccessor.setValue(tc, vaccessor.getValue(tc) + cw * velc);
                    locks[x+bound][y+bound][z+bound].unlock();
                }
            }
        }
    }
}


void extrapolate(openvdb::Vec3dGrid::Ptr velocity, openvdb::BoolGrid::Ptr defined, int bound, openvdb::FloatGrid::Ptr solidGrid)
{
    openvdb::Int32Grid::Ptr numTimesDefined = openvdb::Int32Grid::create();
    openvdb::Int32Grid::Accessor ndaccessor = numTimesDefined->getAccessor();
    numTimesDefined->fill(openvdb::CoordBBox(openvdb::Coord(-1 * bound), openvdb::Coord(bound)), 0);
    numTimesDefined->tree().voxelizeActiveTiles();
    openvdb::BoolGrid::Accessor daccessor = defined->getAccessor();
    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    std::vector<openvdb::Coord> definedCoord;
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord c = openvdb::Coord(x, y, z);
                    if (daccessor.getValue(c) && !isSolid(c, solidGrid) && isWithinBounds(c, 13))
                    {
                        int minx = c.x() == -1*bound ? c.x() : c.x() - 1;
                        int miny = c.y() == -1*bound ? c.y() : c.y() - 1;
                        int minz = c.z() == -1*bound ? c.z() : c.z() - 1;

                        int maxx = c.x() == bound ? c.x() : c.x() + 1;
                        int maxy = c.y() == bound ? c.y() : c.y() + 1;
                        int maxz = c.z() == bound ? c.z() : c.z() + 1;

                        for (int i = minx; i <= maxx; ++i)
                        {
                            for (int j = miny; j <= maxy; ++j)
                            {
                                for (int k = minz; k <= maxz; ++k)
                                {
                                    openvdb::Coord neighbour = openvdb::Coord(i, j, k);
                                    if (!daccessor.getValue(neighbour))
                                    {
                                        if (ndaccessor.getValue(neighbour) == 0)
                                        {
                                            definedCoord.push_back(neighbour);
                                        }
                                        ndaccessor.setValue(neighbour, ndaccessor.getValue(neighbour) + 1);
                                        vaccessor.setValue(neighbour, vaccessor.getValue(c) + vaccessor.getValue(neighbour));
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }

    for (int i = 0; i < definedCoord.size(); ++i)
    {
        vaccessor.setValue(definedCoord[i], vaccessor.getValue(definedCoord[i])/ndaccessor.getValue(definedCoord[i]));
        daccessor.setValue(definedCoord[i], true);
    }
    while(definedCoord.size()) 
    {
        std::vector<openvdb::Coord> tempCoord;
        for (int iter = 0; iter < definedCoord.size(); ++iter)
        {
            openvdb::Coord c = definedCoord[iter];
            int minx = c.x() == -1*bound ? c.x() : c.x() - 1;
            int miny = c.y() == -1*bound ? c.y() : c.y() - 1;
            int minz = c.z() == -1*bound ? c.z() : c.z() - 1;

            int maxx = c.x() == bound ? c.x() : c.x() + 1;
            int maxy = c.y() == bound ? c.y() : c.y() + 1;
            int maxz = c.z() == bound ? c.z() : c.z() + 1;

            for (int i = minx; i <= maxx; ++i)
            {
                for (int j = miny; j <= maxy; ++j)
                {
                    for (int k = minz; k <= maxz; ++k)
                    {
                        openvdb::Coord neighbour = openvdb::Coord(i, j, k);
                        if (!daccessor.getValue(neighbour))
                        {
                            if (ndaccessor.getValue(neighbour) == 0)
                            {
                                tempCoord.push_back(neighbour);
                            }
                            ndaccessor.setValue(neighbour, ndaccessor.getValue(neighbour) + 1);
                            vaccessor.setValue(neighbour, vaccessor.getValue(c) + vaccessor.getValue(neighbour));
                        }
                    }
                }
            }
        }
        for (int i = 0; i < tempCoord.size(); ++i)
        {
            vaccessor.setValue(tempCoord[i], vaccessor.getValue(tempCoord[i])/ndaccessor.getValue(tempCoord[i]));
            daccessor.setValue(tempCoord[i], true);
        }
        definedCoord.clear();
        definedCoord = tempCoord;
    } 
}

struct PointList
{
    std::vector<openvdb::Vec3d> positions;
    std::vector<openvdb::Vec3d> velocities;
    std::vector<Eigen::Matrix3d> FEvec;
    std::vector<Eigen::Matrix3d> FPvec;
    std::vector<Eigen::Matrix3d> gradV;
    std::vector<double> volume;
    std::vector<bool> initializedVelocities;
    openvdb::FloatGrid::Ptr weights;
    openvdb::BoolGrid::Ptr defined;
    std::mutex ***locks;
    std::map<std::pair<int, int>, Eigen::Matrix3d> mapMatrix;
    int boundary;
    PointList(){};

    void populateMatrices(Eigen::SparseMatrix<double> &A, Eigen::VectorXd &b, double beta, double dt, int numActive, openvdb::Int32Grid::Ptr indices, openvdb::FloatGrid::Ptr solidGrid, openvdb::Vec3dGrid::Ptr gridForces, openvdb::FloatGrid::Ptr fluid, openvdb::Vec3dGrid::Ptr velocity, openvdb::Vec3d gravity)
    {
        openvdb::FloatGrid::Accessor faccessor = fluid->getAccessor();
        openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
        openvdb::Vec3dGrid::Accessor gaccessor = gridForces->getAccessor();
        openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();
        std::vector<Triplet> tripletList;
        double maxForceCoeff = 0;
        double maxForceCoeff2 = 0;
        double maxMi = 0;
        bool yes = false;
        openvdb::Vec3d Force;
        // Again hardcoded according to boundary
        for (int x = -15; x <= 15; ++x)
        {
            for (int y = -15; y <= 15; ++y)
            {
                for (int z = -15; z <= 15; ++z)
                {
                    openvdb::Coord xi(x, y, z);
                    double mi = faccessor.getValue(xi);
                    int k = indaccessor.getValue(xi);
                    if (mi > 0.1)
                    {
                        yes = true;
                        openvdb::Vec3d v = vaccessor.getValue(xi);
                        openvdb::Vec3d f = gaccessor.getValue(xi);
                        double maxf = std::max(fabs(f.x()), std::max(fabs(f.y()), fabs(f.z())));
                        if (maxf > maxForceCoeff)
                        {
                            maxForceCoeff = maxf;
                        }
                        if (maxf/mi > maxForceCoeff2)
                        {
                            maxForceCoeff2 = maxf/mi;
                            maxMi = mi;
                            Force = dt*f/mi;
                        }

                        Eigen::Vector3d vs(v.x() + dt*((1.0/mi)*f.x() + gravity.x()), v.y() + dt*((1.0/mi)*f.y() + gravity.y()), v.z() + dt*((1.0/mi)*f.z()+gravity.z()));
                        b[k*3] = vs(0);
                        b[k*3+1] = vs(1);
                        b[k*3+2] = vs(2);
                    }
                }
            }
        }
        std::cout << "Max Force " << Force << " " << maxMi << " " << maxForceCoeff2 << " " << yes << std::endl;
        for(std::map<std::pair<int, int>, Eigen::Matrix3d>::iterator it = mapMatrix.begin(); it != mapMatrix.end(); it++)
        {
            Eigen::Matrix3d mt;
            int i = it->first.first;
            int j = it->first.second;
            if (i == j)
            {
                mt = Eigen::Matrix3d::Identity() + beta*dt*dt*(it->second);
            }
            else
            {
                mt = beta*dt*dt*(it->second);
            }
            for (int p = 0; p < 3; ++p)
            {
                for (int q = 0; q < 3; ++q)
                {
                    tripletList.push_back(Triplet(i*3+p, j*3+q, mt(p,q)));
                }
            }

        }
        std::cout << "GARR" << std::endl;
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        std::cout << "after" << std::endl;

    }
    void initialize(int bound, openvdb::math::Transform::Ptr trans)
    {
        weights = openvdb::FloatGrid::create();
        weights->setTransform(trans);
        boundary = bound;
        weights->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), 0);
        weights->tree().voxelizeActiveTiles();

        defined = openvdb::BoolGrid::create();
        defined->setTransform(trans);
        defined->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), false);
        defined->tree().voxelizeActiveTiles();

        locks = new std::mutex**[2*boundary+1];
        for (int i = 0; i < (2*boundary+1); ++i)
        {
            locks[i] = new std::mutex*[2*boundary+1];
            for (int j = 0; j < (2*boundary+1); ++j)
            {
                locks[i][j] = new std::mutex[(2*boundary+1)*(2*boundary+1)];
            }
        }
    }
    openvdb::Index64 size() const { return openvdb::Index64(positions.size()); }

    // Bad approach used here
    void add(const openvdb::Vec3d &p) 
    { 
        if(fabs(p.x()) < boundary - 2 && fabs(p.y()) < boundary - 2 && fabs(p.z()) < boundary - 2) 
        {
            positions.push_back(p); 
            // if (p.y() > -11)
            // {
            //     velocities.push_back(openvdb::Vec3d(0, 20, 0)); 
            // }
            // else
            // {
            //     velocities.push_back(openvdb::Vec3d(0, 0, 0)); 
            // }
            velocities.push_back(openvdb::Vec3d(0, -50, 0)); 
            initializedVelocities.push_back(false); 
            FEvec.push_back(Eigen::Matrix3d::Identity()); 
            FPvec.push_back(Eigen::Matrix3d::Identity());
            gradV.push_back(Eigen::Matrix3d::Zero());
            volume.push_back(0.0);
        }
    }

    void updateDeformationGradient(openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr solidGrid, double dt, double dx, double thetac, double thetas)
    {
        double minv = 1 - thetac;
        double maxv = 1 + thetas;
        // for (int i = 0; i < size(); ++i)
        // {
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
            openvdb::Vec3d c = positions[i];
            int fcx = round(c.x());
            int fcy = round(c.y());
            int fcz = round(c.z());
            openvdb::Coord fv(fcx, fcy, fcz);
            openvdb::Vec3d fvv(fcx, fcy, fcz);
            int minx = fcx - 1 > -1 * boundary ? fcx - 1 : -1 * boundary;
            int miny = fcy - 1 > -1 * boundary ? fcy - 1 : -1 * boundary;
            int minz = fcz - 1 > -1 * boundary ? fcz - 1 : -1 * boundary;
            int maxx = fcx + 1 < boundary ? fcx + 1 : boundary;
            int maxy = fcy + 1 < boundary ? fcy + 1 : boundary;
            int maxz = fcz + 1 < boundary ? fcz + 1 : boundary;
            gradV[i] = Eigen::Matrix3d::Zero();
            for (int x = minx; x <= maxx; ++x)
            {
                for (int y = miny; y <= maxy; ++y)
                {
                    for (int z = minz; z <= maxz; ++z)
                    {
                        openvdb::Coord tc = openvdb::Coord(x, y, z);
                        if (!isSolid(tc, solidGrid))
                        {
                            // Eigen::Vector3d gradSpline = getGradW(tc, c);
                            Eigen::Vector3d gradSpline = getGradW(tc, c);
                            openvdb::Vec3d vel = vaccessor.getValue(tc);
                            Eigen::Vector3d evel(vel.x(), vel.y(), vel.z());
                            Eigen::Matrix3d gradient = evel * gradSpline.transpose();
                            // Eigen::Matrix3d gradient = getGradV(tc, p, dx, velocity);
                            gradV[i] += gradient;
                        }
                    }
                }
            }

        });

        // }
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
        // for (int i = 0; i < size(); ++i)
        // {
            Eigen::Matrix3d tFE = (Eigen::Matrix3d::Identity() + dt * gradV[i]) * FEvec[i];
            Eigen::Matrix3d F =  tFE * FPvec[i];
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(tFE, Eigen::ComputeThinU | Eigen::ComputeThinV);
            // Eigen::Matrix3d singular = svd.singularValues();
            Eigen::Vector3d singular = svd.singularValues();
            singular(0) = singular(0) > minv ? singular(0) : minv;
            singular(0) = singular(0) < maxv ? singular(0) : maxv;
            singular(1) = singular(1) > minv ? singular(1) : minv;
            singular(1) = singular(1) < maxv ? singular(1) : maxv;
            singular(2) = singular(2) > minv ? singular(2) : minv;
            singular(2) = singular(2) < maxv ? singular(2) : maxv;
            FEvec[i] = svd.matrixU() * singular.asDiagonal() * svd.matrixV().transpose();
            FPvec[i] = svd.matrixV() * singular.asDiagonal().inverse() * svd.matrixU().transpose() * F;

        });
        // }
        double maxGrad = 0;
        double maxFe = 0;
        double maxFp = 0;
        for (int i = 0; i < size(); ++i)
        {
            double t = std::max(gradV[i].maxCoeff(), -1*gradV[i].minCoeff());
            double t2 = FPvec[i].determinant();
            double t3 = FEvec[i].determinant();
            if (t2 < 0)
            {
                std::cout << "FP determinant negative!!!" << std::endl;
            }
            if (maxGrad < t)
            {
                maxGrad = t;
            }
            if (maxFp < t2)
            {
                maxFp = t2;
            }
            if (maxFe < t3)
            {
                maxFe = t3;
            }
        }
        std::cout << "MAX " << maxGrad << " " << maxFp << " " << maxFe << std::endl;

    }

    void populateGridForces(openvdb::Vec3dGrid::Ptr gridForces, openvdb::Int32Grid::Ptr indices, openvdb::FloatGrid::Ptr solidGrid, openvdb::FloatGrid::Ptr fluid, int numActive, double mu, double lambda, double epsilon, double dx)
    {
        std::cout << "GAH" << std::endl;
        mapMatrix.clear();
        openvdb::Vec3dGrid::Accessor gaccessor = gridForces->getAccessor();
        openvdb::FloatGrid::Accessor faccessor = fluid->getAccessor();
        openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();

        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
        // for (int i = 0; i < size(); ++i)
        // {
            // Eigen::JacobiSVD<Eigen::Matrix3d> svd(FEvec[i], Eigen::ComputeThinU | Eigen::ComputeThinV);

            openvdb::Vec3d c = positions[i];
            int fcx = round(c.x());
            int fcy = round(c.y());
            int fcz = round(c.z());
            openvdb::Coord fv(fcx, fcy, fcz);
            openvdb::Vec3d fvv(fcx, fcy, fcz);

            int minx = fcx - 1 > -1 * boundary ? fcx - 1 : -1 * boundary;
            int miny = fcy - 1 > -1 * boundary ? fcy - 1 : -1 * boundary;
            int minz = fcz - 1 > -1 * boundary ? fcz - 1 : -1 * boundary;
            int maxx = fcx + 1 < boundary ? fcx + 1 : boundary;
            int maxy = fcy + 1 < boundary ? fcy + 1 : boundary;
            int maxz = fcz + 1 < boundary ? fcz + 1 : boundary;

            Eigen::Matrix3d sigma = getSigma(mu, lambda, epsilon, FEvec[i], FPvec[i]);
            for (int x = minx; x <= maxx; ++x)
            {
                for (int y = miny; y <= maxy; ++y)
                {
                    for (int z = minz; z <= maxz; ++z)
                    {
                        openvdb::Coord tc = openvdb::Coord(x, y, z);
                        if (!isSolid(tc, solidGrid))
                        {
                            double mi = faccessor.getValue(tc);
                            Eigen::Vector3d gradSpline = getGradW(tc, c);
                            // Eigen::Vector3d gradSpline = getGradW(tc, fvv);
                            Eigen::Vector3d eforce = -1*volume[i]*sigma*gradSpline;
                            // double maxef = std::max(fabs(eforce.x()), std::max())
                            openvdb::Vec3d force(eforce(0), eforce(1), eforce(2));
                            // openvdb::Vec3d wipj(0, -1.0*mi*gravity.y()*spline(tc.x() - c.x())*spline(tc.y() - c.y())*(spline(tc.z() - c.z())), 0);
                            // Eigen::Vector3d eterm2 = -1.0*mi*gravity.y()*(tc.y()+15)*gradSpline;
                            // Eigen::Vector3d eterm2 = -1.0*mi*gravity.y()*(c.y()+15)*gradSpline;
                            // openvdb::Vec3d term2(eterm2(0), eterm2(1), eterm2(2));
                            locks[x+boundary][y+boundary][z+boundary].lock();
                            // gaccessor.setValue(tc, gaccessor.getValue(tc) + force + term2);
                            gaccessor.setValue(tc, gaccessor.getValue(tc) + force);
                            locks[x+boundary][y+boundary][z+boundary].unlock();
                        }
                    }
                }
            }
            });
        // }
        double maxsigma = 0;
        for (int i = 0; i < size(); ++i)
        {
            // Eigen::Matrix3d sigmat = getSigma(mu, lambda, epsilon, FEvec[i], FPvec[i]);
            // maxsigma = std::max(maxsigma, sigmat.maxCoeff());
            // maxsigma = std::max(maxsigma, -1* sigmat.minCoeff());
            openvdb::Vec3d c = positions[i];
            int fcx = round(c.x());
            int fcy = round(c.y());
            int fcz = round(c.z());

            int minx = fcx - 1 > -1 * boundary ? fcx - 1 : -1 * boundary;
            int miny = fcy - 1 > -1 * boundary ? fcy - 1 : -1 * boundary;
            int minz = fcz - 1 > -1 * boundary ? fcz - 1 : -1 * boundary;
            int maxx = fcx + 1 < boundary ? fcx + 1 : boundary;
            int maxy = fcy + 1 < boundary ? fcy + 1 : boundary;
            int maxz = fcz + 1 < boundary ? fcz + 1 : boundary;

            for (int x = minx; x <= maxx; ++x)
            {
                for (int y = miny; y <= maxy; ++y)
                {
                    for (int z = minz; z <= maxz; ++z)
                    {
                        openvdb::Coord xi = openvdb::Coord(x, y, z);
                        double mi = faccessor.getValue(xi);
                        if (!isSolid(xi, solidGrid) && mi > 0.1)
                        {
                            for (int x2 = minx; x2 <= maxx; ++x2)
                            {
                                for (int y2 = miny; y2 <= maxy; ++y2)
                                {
                                    for (int z2 = minz; z2 <= maxz; ++z2)
                                    {
                                        openvdb::Coord xj = openvdb::Coord(x2, y2, z2);
                                        if (!isSolid(xj, solidGrid) && faccessor.getValue(xj) > 0.1)
                                        {
                                            // hardcoded according to solid boundaries
                                            int indexi = indaccessor.getValue(xi);
                                            int indexj = indaccessor.getValue(xj);
                                            std::pair<int, int> tp = std::make_pair(indexi, indexj);
                                            if (mapMatrix.find(tp) == mapMatrix.end())
                                            {
                                                mapMatrix[tp] = Eigen::Matrix3d::Zero();
                                            }
                                            mapMatrix[tp] += (1.0/mi)*volume[i]*getdPsydx2(xi, xj, c, FEvec[i], FPvec[i], lambda, mu, epsilon);

                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
                    std::cout << "DAH" << std::endl;

    }
    void updateVelocity(Eigen::VectorXd ans, openvdb::Int32Grid::Ptr indices, openvdb::Vec3dGrid::Ptr velocity, openvdb::Vec3dGrid::Ptr gridForces, double dt, openvdb::Vec3d gravity, openvdb::FloatGrid::Ptr fluid, openvdb::FloatGrid::Ptr solidGrid)
    {
        openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
        openvdb::Vec3dGrid::Accessor gaccessor = gridForces->getAccessor();
        openvdb::FloatGrid::Accessor faccessor = fluid->getAccessor();
        openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();
        openvdb::Vec3d zero(0,0,0);
        for(int x = -15; x <=15; x++)
        {
            for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord c = openvdb::Coord(x, y, z);
                    if (!isSolid(c, solidGrid))
                    {
                        // vaccessor.setValue(c, vaccessor.getValue(c) + dt * (gaccessor.getValue(c) / faccessor.getValue(c)));
                        if (faccessor.getValue(c) > 0.1)
                        {
                            int index = indaccessor.getValue(c);
                            vaccessor.setValue(c, openvdb::Vec3d(ans(3*index), ans(3*index+1), ans(3*index+2)));
                        }
                        else
                        {
                            vaccessor.setValue(c, openvdb::Vec3d(0,0,0));
                        }
                        // vaccessor.setValue(c, vaccessor.getValue(c) + dt * (gravity - gaccessor.getValue(c)));
                        // vaccessor.setValue(c, vaccessor.getValue(c) + dt * (gravity));
                    }
                }
            }
        }        
    }

    void findVolume(openvdb::FloatGrid::Ptr fluid, openvdb::FloatGrid::Ptr solidGrid)
    {
        openvdb::FloatGrid::Accessor faccessor = fluid->getAccessor();
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            openvdb::Vec3d c = positions[i];
            int fcx = round(c.x());
            int fcy = round(c.y());
            int fcz = round(c.z());

            int minx = fcx - 1 > -1 * boundary ? fcx - 1 : -1 * boundary;
            int miny = fcy - 1 > -1 * boundary ? fcy - 1 : -1 * boundary;
            int minz = fcz - 1 > -1 * boundary ? fcz - 1 : -1 * boundary;
            int maxx = fcx + 1 < boundary ? fcx + 1 : boundary;
            int maxy = fcy + 1 < boundary ? fcy + 1 : boundary;
            int maxz = fcz + 1 < boundary ? fcz + 1 : boundary;

            for (int x = minx; x <= maxx; ++x)
            {
                for (int y = miny; y <= maxy; ++y)
                {
                    for (int z = minz; z <= maxz; ++z)
                    {
                        openvdb::Coord tc = openvdb::Coord(x, y, z);
                        if (!isSolid(tc, solidGrid))
                        {
                            volume[i] += faccessor.getValue(tc) * spline(c.x() - tc.x()) * spline(c.y() - tc.y()) * spline(c.z() - tc.z());
                        }
                    }
                }
            }
            volume[i] = 1.0/volume[i];
        });
    }
    void interpolate(openvdb::FloatGrid::Ptr containerGrid, openvdb::FloatGrid::Ptr solidGrid)
    {
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            openvdb::Vec3d c = positions[i];
            int fcx = round(c.x());
            int fcy = round(c.y());
            int fcz = round(c.z());

            int minx = fcx - 1 > -1 * boundary ? fcx - 1 : -1 * boundary;
            int miny = fcy - 1 > -1 * boundary ? fcy - 1 : -1 * boundary;
            int minz = fcz - 1 > -1 * boundary ? fcz - 1 : -1 * boundary;
            int maxx = fcx + 1 < boundary ? fcx + 1 : boundary;
            int maxy = fcy + 1 < boundary ? fcy + 1 : boundary;
            int maxz = fcz + 1 < boundary ? fcz + 1 : boundary;

            openvdb::FloatGrid::Accessor caccessor = containerGrid->getAccessor();

            for (int x = minx; x <= maxx; ++x)
            {
                for (int y = miny; y <= maxy; ++y)
                {
                    for (int z = minz; z <= maxz; ++z)
                    {

                            openvdb::Coord tc = openvdb::Coord(x, y, z);
                            double cw = spline(c.x() - x) * spline(c.y() - y) * spline(c.z() - z);
                            if (!isSolid(tc, solidGrid) && cw > 0)
                            {
                                locks[x+boundary][y+boundary][z+boundary].lock();
                                caccessor.setValue(tc, caccessor.getValue(tc) + cw);
                                locks[x+boundary][y+boundary][z+boundary].unlock();
                            }
                    }
                }
            }

        });
    }
    void interpFromGrid(openvdb::Vec3dGrid::Ptr velocity, int bound, openvdb::FloatGrid::Ptr fluidGrid)
    {
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            if (!initializedVelocities[i])
            {
                openvdb::Vec3d position = positions[i];
                    openvdb::Vec3d vel = clampedCatmullRom(position, velocity, bound, fluidGrid);
                    velocities[i] = vel;
                    initializedVelocities[i] = true;
            }
        });
    }

    void initializeAllVelocities()
    {
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            initializedVelocities[i] = true;
        });
    }
    void advect(double maxTimeStep, double dx, double &timestep, openvdb::Vec3dGrid::Ptr velocity, int bound, openvdb::FloatGrid::Ptr fluidGrid, openvdb::FloatGrid::Ptr solidGrid)
    {
        double e = 0.5;
        double maxSpeed = 0.0;
        std::mutex l;
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            

                openvdb::Vec3d vel = clampedCatmullRom(positions[i], velocity, bound, fluidGrid);
                
                // positions[i] = position;
                velocities[i] = vel;
                if (maxSpeed < velocities[i].length())
                {
                    l.lock();
                    if (maxSpeed < velocities[i].length())
                    {
                        maxSpeed = velocities[i].length();
                    }
                    l.unlock();
                }
        });
        if (maxSpeed != 0)
        {
            timestep = maxTimeStep < dx/maxSpeed ? maxTimeStep : dx/maxSpeed;
        }
        else
        {
            timestep = maxTimeStep;
        }

        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            openvdb::Vec3d position = positions[i];
            position += timestep * velocities[i];
            int rx = round(position.x());
            int ry = round(position.y());
            int rz = round(position.z());

            openvdb::Coord rcoord = openvdb::Coord(rx, ry, rz);
            if (isSolid(rcoord, solidGrid))
            {

                openvdb::Vec3d vx = openvdb::Vec3d(velocities[i].x() * timestep, 0, 0);
                openvdb::Vec3d vy = openvdb::Vec3d(0, velocities[i].y() * timestep, 0);
                openvdb::Vec3d vz = openvdb::Vec3d(0, 0, velocities[i].z() * timestep);

                if (isSolid(openvdb::Coord(round((positions[i]+vx).x()), positions[i].y(), positions[i].z()), solidGrid))
                {
                    velocities[i].x() *= -1.0*e;
                    // velocities[i].x() *= 0;
                }
                if (isSolid(openvdb::Coord(positions[i].x(), round((positions[i]+vy).y()), positions[i].z()), solidGrid))
                {
                    velocities[i].y() *= -1.0*e;
                    // velocities[i].y() *= 0;
                }
                if (isSolid(openvdb::Coord(positions[i].x(), positions[i].y(), round((positions[i]+vz).z())), solidGrid))
                {
                    velocities[i].z() *= -1.0*e;
                    // velocities[i].z() *= 0;
                }
                // positions[i] += velocities[i]*timestep;
                positions[i] += velocities[i]*timestep;                
            }
            else
            {
                positions[i] = position;
            }
            }
        );
    }
    void FLIPadvect(double maxTimeStep, double dx, double &timestep, openvdb::Vec3dGrid::Ptr velocity, int bound, openvdb::Vec3dGrid::Ptr velBeforeUpdate, openvdb::FloatGrid::Ptr fluidGrid, openvdb::FloatGrid::Ptr solidGrid, openvdb::Vec3dGrid::Ptr normals)
    {
        double e = 0.0;
        openvdb::Vec3dGrid::Accessor naccessor = normals->getAccessor();
        double maxSpeed = 0.0;
        std::mutex l;
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            openvdb::Vec3d vel = CatmullRomFLIP(positions[i], velocities[i], velocity, bound, velBeforeUpdate, fluidGrid, solidGrid);
            velocities[i] += vel;
            if (maxSpeed < velocities[i].length())
            {
                l.lock();
                if (maxSpeed < velocities[i].length())
                {
                    maxSpeed = velocities[i].length();
                }
                l.unlock();
            }
        });
        if (maxSpeed != 0)
        {
            timestep = maxTimeStep < dx/maxSpeed ? maxTimeStep : dx/maxSpeed;
        }
        else
        {
            timestep = maxTimeStep;
        }
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            openvdb::Vec3d position = positions[i];
            position += timestep * velocities[i];

            // Code to take care of stuck particles
            int rx = position.x() > 0 ? ceil(position.x()) : floor(position.x());
            int ry = position.y() > 0 ? ceil(position.y()) : floor(position.y());
            int rz = position.z() > 0 ? ceil(position.z()) : floor(position.z());

            openvdb::Coord rcoord = openvdb::Coord(rx, ry, rz);
            if (isSolid(rcoord, solidGrid))
            {
                if (isSolid(openvdb::Coord(rx, positions[i].y(), positions[i].z()), solidGrid))
                {
                    velocities[i].x() *= -1.0*e;
                }
                if (isSolid(openvdb::Coord(positions[i].x(), ry, positions[i].z()), solidGrid))
                {
                    velocities[i].y() *= -1.0*e;
                }
                if (isSolid(openvdb::Coord(positions[i].x(), positions[i].y(), rz), solidGrid))
                {
                    velocities[i].z() *= -1.0*e;
                }
                positions[i] += velocities[i]*timestep;                
            }
            else
            {
                positions[i] = position;
            }
            }
);

    }

    void P2Gtransfer(openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr solidGrid, openvdb::FloatGrid::Ptr fluid)
    {
        weights->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), 0);
        defined->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), false);
        openvdb::BoolGrid::Accessor daccessor = defined->getAccessor();
        openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
        for(int x = -15; x <=15; x++)
        {
            for(int y = -15; y <=15; y++)
                {
                    for (int z = -15; z <= 15; z++)
                    {
                        openvdb::Coord c = openvdb::Coord(x, y, z);
                        if (fabs(c.x()) > 13 || fabs(c.y()) > 13 || fabs(c.z()) > 13 || isSolid(c, solidGrid))
                        {
                            daccessor.setValue(c, true);
                        }
                    }
                }
        }
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
                p2gCatmullRom(positions[i], velocities[i], velocity, weights, boundary, locks, solidGrid);
        });
        // openvdb::FloatGrid::Accessor waccessor = weights->getAccessor();
        openvdb::FloatGrid::Accessor waccessor = fluid->getAccessor();
        for(int x = -15; x <=15; x++)
        {
            for(int y = -15; y <=15; y++)
                {
                    for (int z = -15; z <= 15; z++)
                    {
                        openvdb::Coord c = openvdb::Coord(x, y, z);
                        double w = waccessor.getValue(c);
                        if (w > 0.1)
                        {
                            vaccessor.setValue(c, vaccessor.getValue(c)/w);
                            daccessor.setValue(c, true);
                        }
                        else
                        {
                            vaccessor.setValue(c, openvdb::Vec3d(0,0,0));
                        }
                    }
                }
        }
        // extrapolate(velocity, defined, boundary, solidGrid);
    }
};

int main(int argc, char* argv[])
{
    openvdb::initialize();
    openvdb::math::Transform::Ptr trans = openvdb::math::Transform::createLinearTransform(factor);

    // The container grid which contains the fluid
    openvdb::FloatGrid::Ptr containerGrid = openvdb::FloatGrid::create(0);
    containerGrid->setTransform(trans);
    containerGrid->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);

    openvdb::FloatGrid::Ptr outputGrid = openvdb::FloatGrid::create(0);
    outputGrid->setTransform(trans);
    outputGrid->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);
    outputGrid->tree().voxelizeActiveTiles();

    openvdb::FloatGrid::Ptr solidGrid = openvdb::FloatGrid::create();
    solidGrid->setTransform(trans);
    solidGrid->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0);

    // The fluid grid represents the fluid
    openvdb::FloatGrid::Ptr fluidGrid = openvdb::FloatGrid::create(-1);
    fluidGrid->setTransform(trans);


    // For normal fluid
    // fluidGrid->fill(openvdb::CoordBBox(openvdb::Coord(-13), openvdb::Coord(-10)), 0, true);

    // For pea fluid
    // fluidGrid->fill(openvdb::CoordBBox(openvdb::Coord(-1), openvdb::Coord(1)), 0, true);
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -1; i <= 2; ++i)
    // {
    //     for (int j = -13; j <= -10; ++j)
    //     {
    //         for (int k = -1; k <= 2; ++k)
    //         {
    //             fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //         }
    //     }
    // }

    // For cone
    openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    for (int i = -13; i <= 13; ++i)
    {
        for (int j = -13; j <= -10; ++j)
        {
            for (int k = -13; k <= 13; ++k)
            {
                double r = (double)(j+13)/2;
                if (i*i + k*k <= r*r)
                {
                    fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
                }
            }
        }
    }

    // For double balls
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -13; i <= 13; ++i)
    // {
    //     for (int j = -13; j <= 13; ++j)
    //     {
    //         for (int k = -13; k <= 13; ++k)
    //         {
    //             // double r = (double)(j+13)/2;
    //             if ((i)*(i) + k*k + (j+11)*(j+11) <= 2*2)
    //             {
    //                 fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //             }
    //         }
    //     }
    // }
    // for (int i = -13; i <= 13; ++i)
    // {
    //     for (int j = -13; j <= 13; ++j)
    //     {
    //         for (int k = -13; k <= 13; ++k)
    //         {
    //             // double r = (double)(j+13)/2;
    //             if ((i)*(i) + k*k + (j+7)*(j+7) <= 2*2)
    //             {
    //                 fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //             }
    //         }
    //     }
    // }
    // For sphere
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -13; i <= 13; ++i)
    // {
    //     for (int j = -13; j <= 13; ++j)
    //     {
    //         for (int k = -13; k <= 13; ++k)
    //         {
    //             int jt = j+10;
    //             if (i*i + k*k + jt*jt <= 9)
    //             {
    //                 fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //             }
    //         }
    //     }
    // }
    // For O
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -13; i <= 13; ++i)
    // {
    //     for (int j = -13; j <= 13; ++j)
    //     {
    //         for (int k = 0; k <= 0; ++k)
    //         {
    //             double it = i;
    //             double jt = j+8;
    //             if (it*it + jt*jt <= 25 && it*it + jt*jt >= 16)
    //             {
    //                 fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //             }
    //         }
    //     }
    // }
    // For side fluid
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -57; i <= 57; ++i)
    // {
    //     for (int j = -57; j <= -40; ++j)
    //     {
    //         for (int k = -57; k <= -40; ++k)
    //         {
    //             fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //         }
    //     }
    // }

    // For stable fluid
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -57; i <= 57; ++i)
    // {
    //     for (int j = -57; j <= -55; ++j)
    //     {
    //         for (int k = -57; k <= 57; ++k)
    //         {
    //             fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //         }
    //     }
    // }

    openvdb::Int32Grid::Ptr indices = openvdb::Int32Grid::create(0);
    indices->setTransform(trans);

    openvdb::Vec3dGrid::Ptr vels = openvdb::Vec3dGrid::create();
    vels->setTransform(trans);
    vels->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), openvdb::Vec3d(0, 0, 0), true);

    openvdb::Vec3dGrid::Ptr normals = openvdb::Vec3dGrid::create(openvdb::Vec3d(0, 0, 0));
    normals->setTransform(trans);
    normals->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), openvdb::Vec3d(0, 0, 0), true);

    openvdb::FloatGrid::Accessor outGridAccessor = outputGrid->getAccessor();
    openvdb::Vec3dGrid::Accessor normalAccessor = normals->getAccessor();
    openvdb::FloatGrid::Accessor saccessor = solidGrid->getAccessor();
    openvdb::Vec3dGrid::Accessor vaccessor = vels->getAccessor();
    openvdb::FloatGrid::Accessor caccessor = containerGrid->getAccessor();

    using Randgen = std::mersenne_twister_engine<uint32_t, 32, 351, 175, 19, 0xccab8ee7, 11, 0xffffffff, 7, 0x31b6ab00, 15, 0xffe50000, 17, 1812433253>;
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {

                    openvdb::Coord xyz = openvdb::Coord(x, y, z);
                    if (fabs(xyz.x()) > 13 || fabs(xyz.y()) > 13 || fabs(xyz.z()) > 13)
                    {
                        saccessor.setValue(xyz, 1);
                        openvdb::Vec3d vec = normalAccessor.getValue(xyz);
                        if (fabs(xyz.x()) > 13)
                        {
                            if (xyz.x() < 0)
                            {
                                normalAccessor.setValue(xyz, openvdb::Vec3d(1, vec.y(), vec.z()));
                            }
                            else
                            {
                                normalAccessor.setValue(xyz, openvdb::Vec3d(-1, vec.y(), vec.z()));
                            }
                        }
                        if (fabs(xyz.y()) > 13)
                        {
                            if (xyz.y() < 0)
                            {
                                normalAccessor.setValue(xyz, openvdb::Vec3d(vec.x(), 1, vec.z()));
                            }
                            else
                            {
                                normalAccessor.setValue(xyz, openvdb::Vec3d(vec.x(), -1, vec.z()));                    
                            }
                        }
                        if (fabs(xyz.z()) > 13)
                        {
                            if (xyz.z() < 0)
                            {
                                normalAccessor.setValue(xyz, openvdb::Vec3d(vec.x(), vec.y(), 1));
                            }
                            else
                            {
                                normalAccessor.setValue(xyz, openvdb::Vec3d(vec.x(), vec.y(), -1));                    
                            }
                        }
                    }

        // for two blocks
        // if((xyz.x() >= -11 && xyz.x() <= -6) || (xyz.x() <= 11 && xyz.x() >= 6))
        // {
        //     if((xyz.z() >= -3 && xyz.z() <= 3))
        //     {
        //         if((xyz.y() >= -13 && xyz.y() <= -8))
        //         {
        //             iter.setValue(1);
        //             outGridAccessor.setValue(xyz, 1);
        //         }
        //     }
        // }

        // for three blocks
        // if((xyz.x() >= -11 && xyz.x() <= -7) || (xyz.x() >= -2 && xyz.x() <= 2) || (xyz.x() <= 11 && xyz.x() >= 7))
        // {
        //     if((xyz.z() >= -3 && xyz.z() <= 3))
        //     {
        //         if((xyz.y() >= -13 && xyz.y() <= -8))
        //         {
        //             iter.setValue(1);
        //             outGridAccessor.setValue(xyz, 1);
        //         }
        //     }
        // }

                }
            }
    }

    // for big wall
    // for (int i = -13; i <= 13; ++i)
    // {
    //     for (int j = -13; j <= -50; ++j)
    //     {
    //         for (int k = -15; k <= -25; ++k)
    //         {
    //             openvdb::Coord xyz = openvdb::Coord(i, j, k);
    //             saccessor.setValue(xyz, 1);
    //             outGridAccessor.setValue(xyz, 1);
    //         }
    //     }
    // }
    // openvdb::GridPtrVec grids;
    PointList pos;
    pos.initialize(15, trans);
    // const openvdb::Index64 pointCount = 400;
    std::mt19937 mtRandi(0);
    openvdb::tools::UniformPointScatter<PointList, std::mt19937> scatteri(pos, 400.f, mtRandi);
    scatteri.operator()<openvdb::FloatGrid>(*fluidGrid);

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double>> cg;

    openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();
    bool isIndiceVoxelized = false;
    bool areVoxelized = false;
    openvdb::Vec3d gravity(0, -10, 0);
    double dx = 1.0*factor;
    openvdb::Vec3dGrid::Ptr velBeforeUpdate;

    pos.initializeAllVelocities();
    openvdb::BoolGrid::Ptr defined = pos.defined -> deepCopy();
    double simulationTime = 0;
    std::string filename = "mygrids.vdb";
    openvdb::io::File file(filename);
    openvdb::GridPtrVec grids;
    double dt = 0.001;
    openvdb::Vec3dGrid::Ptr gridForces = openvdb::Vec3dGrid::create(openvdb::Vec3d(0, 0, 0));
    gridForces->setTransform(trans);
for (int i = 0; i < 500; ++i)
{
    // open file
    std::string filename = "simulation/mygrids" + std::to_string(i) + ".vdb";
    openvdb::io::File file2(filename);
    openvdb::GridPtrVec grids2;
    std::mt19937 mtRand(i+1);
    // openvdb::tools::UniformPointScatter<PointList, std::mt19937> scatter(pos, 10.f, mtRand);

    
    // if (i%5 == 0)
    // {
        // scatter.operator()<openvdb::FloatGrid>(*fluidGrid);
    // }
    std::cout << "DT " << dt << std::endl;

    indices->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), -1, true);
    gridForces->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), openvdb::Vec3d(0,0,0));
    if (!isIndiceVoxelized)
    {
        indices->tree().voxelizeActiveTiles();
        isIndiceVoxelized = true;
    }

    int numActive = 0;
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord xyz = openvdb::Coord(x, y, z);
                    // if(!isSolid(xyz, solidGrid))
                    // {
                        caccessor.setValue(xyz, 0);
                    // }
                }
            }
    }



    pos.interpolate(containerGrid, solidGrid);
    pos.P2Gtransfer(vels, solidGrid, containerGrid);
    if (i == 0)
    {
        pos.findVolume(containerGrid, solidGrid);
    }

    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord xyz = openvdb::Coord(x, y, z);
                    if(!isSolid(xyz, solidGrid) && isWithinBounds(xyz, 13))
                    {
                        if (caccessor.getValue(xyz) > 0.1)
                        {
                            indaccessor.setValue(xyz, numActive);
                            numActive++;
                        }
                    }
                }
            }
    }
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord xyz = openvdb::Coord(x, y, z);
                    double val = caccessor.getValue(xyz);
                    if (!isSolid(xyz, solidGrid) && val > 0.1)
                    {
                        outGridAccessor.setValue(xyz, val);
                    }
                }
            }
    }
    Eigen::VectorXd vsolve(3*numActive);

    grids.push_back(outputGrid -> deepCopy());
    grids2.push_back(outputGrid -> deepCopy());
    // grids.push_back(vels -> deepCopy());
    // grids2.push_back(vels -> deepCopy());


    // grids.push_back(vels -> deepCopy());
    Eigen::VectorXd b2(numActive);        
    double error;
    velBeforeUpdate = vels -> deepCopy();
    double E = 48000;
    double nu = 0.47;
    double beta = 0.5;
    std::cout << "1" << std::endl;
    pos.populateGridForces(gridForces, indices, solidGrid, containerGrid, numActive, E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu)), 10, dx);
    std::cout << "2" << std::endl;
    Eigen::SparseMatrix<double> A(3*numActive, 3*numActive);    
    Eigen::VectorXd b(3*numActive);        
    pos.populateMatrices(A, b, beta, dt, numActive, indices, solidGrid, gridForces, containerGrid, vels, gravity);
    cg.compute(A);
    vsolve = cg.solve(b);
    pos.updateVelocity(vsolve, indices, vels, gridForces, dt, gravity, containerGrid, solidGrid);
    std::cout << "Error: " << cg.error() << std::endl;
    std::cout << "5" << std::endl;
    // grids.push_back(vels -> deepCopy());
    // grids2.push_back(vels -> deepCopy());
    std::cout << "3" << std::endl;
    pos.updateDeformationGradient(vels, solidGrid, dt, dx, 0.025, 0.0075);
    std::cout << "4" << std::endl;


    // pos.advect(0.1, dx, dt, vels, 15, containerGrid, solidGrid);
    // Same to do for pic
    pos.FLIPadvect(0.001, dx, dt, vels, 15, velBeforeUpdate, containerGrid, solidGrid, normals);
    std::cout << "DT " << dt << std::endl;
    if(i < 500)
    {

        // scatter.operator()<openvdb::FloatGrid>(*fluidGrid);
        // // //containerGrid is dummy here
        // pos.interpFromGrid(vels, 15, containerGrid);
    }
    // std::cout << "Error:\t" << error << std::endl;
    std::cout << "Iteration:\t" << i+1 << std::endl;
    simulationTime += dt;
    vels->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), openvdb::Vec3d(0, 0, 0), true);
    // std::cout << "Time delta:\t" << simulationTime << std::endl;
    // file.write(grids);
    file2.write(grids2);
    file2.close();

}
    file.write(grids);
    file.close();

end = clock();
cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
std::cout << "Time Taken " << cpu_time_used/15 << " minutes" << std::endl; 
}
