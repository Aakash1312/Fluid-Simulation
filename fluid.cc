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
#include <Eigen/IterativeLinearSolvers>

typedef Eigen::Triplet<double> Triplet;

double spline(double x)
{
    if (x < 0)
    {
        x *= -1.0;
    }
    if (x < 1)
    {
        return (0.5*x*x*x - x*x + 2.0/3.0);
    }
    if (x < 2)
    {
        return (-1.0 * (x * x * x)/6.0) + x*x - 2*x + 4.0/3.0;
    }
    return 0;
}

openvdb::Vec3d clampedCatmullRom(openvdb::Vec3d c, openvdb::Vec3dGrid::Ptr velocity, int bound)
{
    int fcx = floor(c.x());
    int fcy = floor(c.y());
    int fcz = floor(c.z());

    int minx = fcx - 3 > -1 * bound ? fcx - 3 : -1 * bound;
    int miny = fcy - 3 > -1 * bound ? fcy - 3 : -1 * bound;
    int minz = fcz - 3 > -1 * bound ? fcz - 3 : -1 * bound;
    int maxx = fcx + 3 < bound ? fcx + 3 : bound;
    int maxy = fcy + 3 < bound ? fcy + 3 : bound;
    int maxz = fcz + 3 < bound ? fcz + 3 : bound;

    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();

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
                velc = vaccessor.getValue(openvdb::Coord(x, y, z));

                minu = velc.x() < minu ? velc.x() : minu;
                minv = velc.y() < minv ? velc.y() : minv;
                minw = velc.z() < minw ? velc.z() : minw;

                maxu = velc.x() > maxu ? velc.x() : maxu;
                maxv = velc.y() > maxv ? velc.y() : maxv;
                maxw = velc.z() > maxw ? velc.z() : maxw;

                double cw = spline(c.x() - x) * spline(c.y() - y) * spline(c.z() - z);
                weight += cw;
                u += velc.x() * cw;
                v += velc.y() * cw;
                w += velc.z() * cw;
            }
        }
    }

    u /= weight;
    v /= weight;
    w /= weight;

    u = u > maxu ? maxu : u;
    v = v > maxv ? maxv : v;
    w = w > maxw ? maxw : w;

    u = u < minu ? minu : u;
    v = v < minv ? minv : v;
    w = w < minw ? minw : w;

    return openvdb::Vec3d(u, v, w);
}

openvdb::Vec3d CatmullRomFLIP(openvdb::Vec3d c, openvdb::Vec3d vel, openvdb::Vec3dGrid::Ptr velocity, int bound)
{
    int fcx = floor(c.x());
    int fcy = floor(c.y());
    int fcz = floor(c.z());

    int minx = fcx - 3 > -1 * bound ? fcx - 3 : -1 * bound;
    int miny = fcy - 3 > -1 * bound ? fcy - 3 : -1 * bound;
    int minz = fcz - 3 > -1 * bound ? fcz - 3 : -1 * bound;
    int maxx = fcx + 3 < bound ? fcx + 3 : bound;
    int maxy = fcy + 3 < bound ? fcy + 3 : bound;
    int maxz = fcz + 3 < bound ? fcz + 3 : bound;

    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    double weight = 0;

    openvdb::Vec3d velc;
    openvdb::Vec3d delta(0, 0, 0);

    for (int x = minx; x <= maxx; ++x)
    {
        for (int y = miny; y <= maxy; ++y)
        {
            for (int z = minz; z <= maxz; ++z)
            {
                velc = vaccessor.getValue(openvdb::Coord(x, y, z));
                double cw = spline(c.x() - x) * spline(c.y() - y) * spline(c.z() - z);
                weight += cw;
                delta += (velc - vel) * cw;
            }
        }
    }

    return delta/weight;
}

void p2gCatmullRom(openvdb::Vec3d c, openvdb::Vec3d velc, openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr weights, int bound)
{
    int fcx = floor(c.x());
    int fcy = floor(c.y());
    int fcz = floor(c.z());

    int minx = fcx - 3 > -1 * bound ? fcx - 3 : -1 * bound;
    int miny = fcy - 3 > -1 * bound ? fcy - 3 : -1 * bound;
    int minz = fcz - 3 > -1 * bound ? fcz - 3 : -1 * bound;
    int maxx = fcx + 3 < bound ? fcx + 3 : bound;
    int maxy = fcy + 3 < bound ? fcy + 3 : bound;
    int maxz = fcz + 3 < bound ? fcz + 3 : bound;

    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::FloatGrid::Accessor waccessor = weights->getAccessor();

    for (int x = minx; x <= maxx; ++x)
    {
        for (int y = miny; y <= maxy; ++y)
        {
            for (int z = minz; z <= maxz; ++z)
            {
                openvdb::Coord tc = openvdb::Coord(x, y, z);
                double cw = spline(c.x() - x) * spline(c.y() - y) * spline(c.z() - z);
                waccessor.setValue(tc, waccessor.getValue(tc) + cw);
                vaccessor.setValue(tc, vaccessor.getValue(tc) + cw * velc);
            }
        }
    }
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

bool isWithinBounds(openvdb::Coord xyz, int bound)
{
    if (abs(xyz.x()) > bound || abs(xyz.y()) > bound || abs(xyz.z()) > bound)
    {
        return false;
    }
    return true;
}

void setA(double dx, double dt, double rho, openvdb::FloatGrid::Ptr grid, openvdb::FloatGrid::Ptr Adiag, openvdb::FloatGrid::Ptr Aplusi, openvdb::FloatGrid::Ptr Aplusj, openvdb::FloatGrid::Ptr Aplusk, openvdb::FloatGrid::Ptr solidGrid)
{
    double scale = dt/(rho * dx * dx);
    openvdb::FloatGrid::Accessor daccessor = Adiag->getAccessor();
    openvdb::FloatGrid::Accessor iaccessor = Aplusi->getAccessor();
    openvdb::FloatGrid::Accessor jaccessor = Aplusj->getAccessor();
    openvdb::FloatGrid::Accessor kaccessor = Aplusk->getAccessor();
    openvdb::FloatGrid::Accessor gaccessor = grid->getAccessor();

    for (openvdb::FloatGrid::ValueOnIter iter = grid->beginValueOn(); iter; ++iter)
    {
        double val = iter.getValue();
        openvdb::Coord c(iter.getCoord());
        openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
        openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
        openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);


        if (!isSolid(c, solidGrid) && val > 0)
        {
                if (!isSolid(ipjk, solidGrid))
                {
                    double val2 = gaccessor.getValue(ipjk);
                    if (val2 > 0)
                    {
                        daccessor.setValue(c, daccessor.getValue(c) + scale);
                        daccessor.setValue(ipjk, daccessor.getValue(ipjk) + scale);
                        iaccessor.setValue(c, -1 * scale);
                    }
                    else
                    {
                        daccessor.setValue(c, daccessor.getValue(c) + scale);
                    }
                }

                if (!isSolid(ijpk, solidGrid))
                {
                    double val2 = gaccessor.getValue(ijpk);
                    if (val2 > 0)
                    {
                        daccessor.setValue(c, daccessor.getValue(c) + scale);
                        daccessor.setValue(ijpk, daccessor.getValue(ijpk) + scale);
                        jaccessor.setValue(c, -1 * scale);
                    }
                    else
                    {
                        daccessor.setValue(c, daccessor.getValue(c) + scale);
                    }
                }


                if (!isSolid(ijkp, solidGrid))
                {
                    double val2 = gaccessor.getValue(ijkp);
                    if (val2 > 0)
                    {
                        daccessor.setValue(c, daccessor.getValue(c) + scale);
                        daccessor.setValue(ijkp, daccessor.getValue(ijkp) + scale);
                        kaccessor.setValue(c, -1 * scale);
                    }
                    else
                    {
                        daccessor.setValue(c, daccessor.getValue(c) + scale);
                    }
                }
        }
    }

}

void setRHS(double dx, openvdb::FloatGrid::Ptr grid, openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr rhs, openvdb::FloatGrid::Ptr solidGrid)
{
    double scale = 1.0/dx;
    openvdb::FloatGrid::Accessor raccessor = rhs->getAccessor();
    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();

    for (openvdb::FloatGrid::ValueOnIter iter = grid->beginValueOn(); iter; ++iter)
    {
        double val = iter.getValue();
        openvdb::Coord c(iter.getCoord());

        // It may in future create segmentation fault as it is not within boundary
        openvdb::Coord imjk(c.x()-1, c.y(), c.z());
        openvdb::Coord ipjk(c.x()+1, c.y(), c.z());

        openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
        openvdb::Coord ijmk(c.x(), c.y()-1, c.z());

        openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);
        openvdb::Coord ijkm(c.x(), c.y(), c.z()-1);

        openvdb::Vec3d v = vaccessor.getValue(c);
        openvdb::Vec3d vi = vaccessor.getValue(ipjk);
        openvdb::Vec3d vj = vaccessor.getValue(ijpk);
        openvdb::Vec3d vk = vaccessor.getValue(ijkp);

        if (val > 0 && !isSolid(c, solidGrid))
        {
            if (isWithinBounds(imjk, 15) && isSolid(imjk, solidGrid))
            {
                raccessor.setValue(c, raccessor.getValue(c) - (scale * v.x()));
            }
            if (isWithinBounds(ipjk, 15) && isSolid(ipjk, solidGrid))
            {
                raccessor.setValue(c, raccessor.getValue(c) + (scale * vi.x()));
            }

            if (isWithinBounds(ijmk, 15) && isSolid(ijmk, solidGrid))
            {
                raccessor.setValue(c, raccessor.getValue(c) - (scale * v.y()));
            }
            if (isWithinBounds(ijpk, 15) && isSolid(ijpk, solidGrid))
            {
                raccessor.setValue(c, raccessor.getValue(c) + (scale * vj.y()));
            }

            if (isWithinBounds(ijkm, 15) && isSolid(ijkm, solidGrid))
            {
                raccessor.setValue(c, raccessor.getValue(c) - (scale * v.z()));
            }
            if (isWithinBounds(ijkp, 15) && isSolid(ijkp, solidGrid))
            {
                raccessor.setValue(c, raccessor.getValue(c) + (scale * vk.z()));
            }

        }
    }

}

void setA2(openvdb::Int32Grid::Ptr indices, Eigen::VectorXd &x, Eigen::SparseMatrix<double> &A, openvdb::FloatGrid::Ptr diver, openvdb::FloatGrid::Ptr Adiag, openvdb::FloatGrid::Ptr Aplusi, openvdb::FloatGrid::Ptr Aplusj, openvdb::FloatGrid::Ptr Aplusk)
{
    std::vector<Triplet> tripletList;
    openvdb::FloatGrid::Accessor iaccessor = Aplusi->getAccessor();
    openvdb::FloatGrid::Accessor jaccessor = Aplusj->getAccessor();
    openvdb::FloatGrid::Accessor kaccessor = Aplusk->getAccessor();
    openvdb::FloatGrid::Accessor daccessor = diver->getAccessor();
    openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();

    for (openvdb::FloatGrid::ValueOnIter iter = Adiag->beginValueOn(); iter; ++iter)
    {
        if (iter.getValue() != 0)
        {
            openvdb::Coord c = iter.getCoord();
            int index = indaccessor.getValue(c);
            openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
            openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
            openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);

            if (iaccessor.getValue(c) != 0)
            {
                tripletList.push_back(Triplet(index, indaccessor.getValue(ipjk), iaccessor.getValue(c)));
            }
            if (jaccessor.getValue(c) != 0)
            {
                tripletList.push_back(Triplet(index, indaccessor.getValue(ijpk), jaccessor.getValue(c)));
            }
            if (kaccessor.getValue(c) != 0)
            {
                tripletList.push_back(Triplet(index, indaccessor.getValue(ijkp), kaccessor.getValue(c)));
            }

            openvdb::Coord imjk(c.x()-1, c.y(), c.z());
            openvdb::Coord ijmk(c.x(), c.y()-1, c.z());
            openvdb::Coord ijkm(c.x(), c.y(), c.z()-1);
            if (iaccessor.getValue(imjk) != 0)
            {
                tripletList.push_back(Triplet(index, indaccessor.getValue(imjk), iaccessor.getValue(imjk)));
            }
            if (jaccessor.getValue(ijmk) != 0)
            {
                tripletList.push_back(Triplet(index, indaccessor.getValue(ijmk), jaccessor.getValue(ijmk)));
            }
            if (kaccessor.getValue(ijkm) != 0)
            {
                tripletList.push_back(Triplet(index, indaccessor.getValue(ijkm), kaccessor.getValue(ijkm)));
            }

            tripletList.push_back(Triplet(index, index, iter.getValue()));
            x(index) = daccessor.getValue(c);
        }
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

void setOnlyB(openvdb::Int32Grid::Ptr indices, Eigen::VectorXd &x, openvdb::FloatGrid::Ptr diver, openvdb::FloatGrid::Ptr Adiag)
{
    openvdb::FloatGrid::Accessor daccessor = diver->getAccessor();
    openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();

    for (openvdb::FloatGrid::ValueOnIter iter = Adiag->beginValueOn(); iter; ++iter)
    {
        if (iter.getValue() != 0)
        {
            openvdb::Coord c = iter.getCoord();
            int index = indaccessor.getValue(c);
            x(index) = daccessor.getValue(c);
        }
    }
}

void setDiver(double dx, openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr diver, openvdb::FloatGrid::Ptr rhs, openvdb::FloatGrid::Ptr fluidGrid, openvdb::FloatGrid::Ptr solidGrid)
{
    openvdb::FloatGrid::Accessor daccessor = diver->getAccessor();
    openvdb::FloatGrid::Accessor raccessor = rhs->getAccessor();
    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    for (openvdb::FloatGrid::ValueOnIter iter = fluidGrid->beginValueOn(); iter; ++iter)
    {
        openvdb::Coord c(iter.getCoord());
        if (iter.getValue() >  0 && !isSolid(c, solidGrid))
        {
            openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
            openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
            openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);

            openvdb::Vec3d v = vaccessor.getValue(c);
            openvdb::Vec3d vi = vaccessor.getValue(ipjk);
            openvdb::Vec3d vj = vaccessor.getValue(ijpk);
            openvdb::Vec3d vk = vaccessor.getValue(ijkp);

            double u = (vi.x() - v.x())/dx;
            double vd = (vj.y() - v.y())/dx;
            double w = (vk.z() - v.z())/dx;
            daccessor.setValue(c, (raccessor.getValue(c)) - u - vd - w);
        }
    }
}

void velUpdate(openvdb::Int32Grid::Ptr indices, double dx, double dt, double rho, openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr fluidGrid, Eigen::VectorXd pressure, openvdb::FloatGrid::Ptr solidGrid)
{
    double scale = dt/(rho * dx);
    openvdb::FloatGrid::Accessor faccessor = fluidGrid->getAccessor();
    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();
    for (openvdb::Vec3dGrid::ValueOnIter iter = velocity->beginValueOn(); iter; ++iter)
    {
        openvdb::Coord c(iter.getCoord());

        // It may in future lead to segmentation fault
        openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
        openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
        openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);

        double val = faccessor.getValue(c);
        if (!isSolid(c, solidGrid))
        {
            if (val > 0)
            {
                double pre = pressure(indaccessor.getValue(c));
                double u = iter.getValue().x() - scale * pre;
                double v = iter.getValue().y() - scale * pre;
                double w = iter.getValue().z() - scale * pre;
                vaccessor.setValue(c, openvdb::Vec3d(u, v, w));
                if (isWithinBounds(ipjk, 15))
                {
                    double up = vaccessor.getValue(ipjk).x() + scale * pre;
                    vaccessor.setValue(ipjk, openvdb::Vec3d(up, vaccessor.getValue(ipjk).y(), vaccessor.getValue(ipjk).z()));
                }

                if (isWithinBounds(ijpk, 15))
                {
                    double vp = vaccessor.getValue(ijpk).y() + scale * pre;
                    vaccessor.setValue(ijpk, openvdb::Vec3d(vaccessor.getValue(ijpk).x(), vp, vaccessor.getValue(ijpk).z()));
                }

                if (isWithinBounds(ijkp, 15))
                {
                    double wp = vaccessor.getValue(ijkp).z() + scale * pre;
                    vaccessor.setValue(ijkp, openvdb::Vec3d(vaccessor.getValue(ijkp).x(), vaccessor.getValue(ijkp).y(), wp));
                }
            }
        }
        else
        {

            vaccessor.setValue(c, openvdb::Vec3d(0, 0, 0));
            if (isWithinBounds(ipjk, 15))
            {
                vaccessor.setValue(ipjk, openvdb::Vec3d(0, vaccessor.getValue(ipjk).y(), vaccessor.getValue(ipjk).z()));
            }
            if (isWithinBounds(ijpk, 15))
            {
                vaccessor.setValue(ijpk, openvdb::Vec3d(vaccessor.getValue(ijpk).x(), 0, vaccessor.getValue(ijpk).z()));
            }
            if (isWithinBounds(ijkp, 15))
            {
                vaccessor.setValue(ijkp, openvdb::Vec3d(vaccessor.getValue(ijkp).x(), vaccessor.getValue(ijkp).y(), 0));
            }
        }
    }
}

void extrapolate(openvdb::Vec3dGrid::Ptr velocity, openvdb::BoolGrid::Ptr defined, int bound)
{
    openvdb::Int32Grid::Ptr numTimesDefined = openvdb::Int32Grid::create();
    openvdb::Int32Grid::Accessor ndaccessor = numTimesDefined->getAccessor();
    numTimesDefined->fill(openvdb::CoordBBox(openvdb::Coord(-1 * bound), openvdb::Coord(bound)), 0);
    numTimesDefined->tree().voxelizeActiveTiles();
    openvdb::BoolGrid::Accessor daccessor = defined->getAccessor();
    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    std::vector<openvdb::Coord> definedCoord;
    for (openvdb::BoolGrid::ValueOnIter iter = defined->beginValueOn(); iter; ++iter)
    {
        if (iter.getValue())
        {
            openvdb::Coord c = iter.getCoord();
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
    openvdb::FloatGrid::Ptr weights;
    openvdb::BoolGrid::Ptr defined;
    int boundary;
    PointList(){};
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
    }
    openvdb::Index64 size() const { return openvdb::Index64(positions.size()); }
    void add(const openvdb::Vec3d &p) { positions.push_back(openvdb::Vec3d(p.x(),p.y(),p.z())); velocities.push_back(openvdb::Vec3d(0, 0, 0));}
    double interpolate(openvdb::Vec3d xyz)
    {
        double sum = 0;
        for (int i = 0; i < size(); ++i)
        {
            sum += spline(positions[i].x() - xyz.x()) * spline(positions[i].y() - xyz.y()) * spline(positions[i].z() - xyz.z());
        }

        return sum - 0.001;
    }

    void advect(double timestep, openvdb::Vec3dGrid::Ptr velocity, int bound)
    {
        for (int i = 0; i < size(); ++i)
        {
            openvdb::Vec3d position = positions[i];
            openvdb::Vec3d vel = clampedCatmullRom(position, velocity, bound);
            position += timestep * vel;
            positions[i] = position;
            velocities[i] = vel;
        }
    }

    void FLIPadvect(double timestep, openvdb::Vec3dGrid::Ptr velocity, int bound)
    {
        for (int i = 0; i < size(); ++i)
        {
            openvdb::Vec3d position = positions[i];
            openvdb::Vec3d vel = CatmullRomFLIP(positions[i], velocities[i], velocity, bound);
            velocities[i] += vel;
            position += timestep * velocities[i];
            positions[i] = position;
        }
    }

    void P2Gtransfer(openvdb::Vec3dGrid::Ptr velocity)
    {
        weights->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), 0);
        defined->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), false);
        openvdb::FloatGrid::Accessor waccessor = weights->getAccessor();
        openvdb::BoolGrid::Accessor daccessor = defined->getAccessor();
        for (int i = 0; i < size(); ++i)
        {
            p2gCatmullRom(positions[i], velocities[i], velocity, weights, boundary);
        }
        for (openvdb::Vec3dGrid::ValueOnIter iter = velocity->beginValueOn(); iter; ++iter)
        {
            double w = waccessor.getValue(iter.getCoord());
            if (w != 0)
            {
                iter.setValue(iter.getValue()/w);
                daccessor.setValue(iter.getCoord(), true);
            }
        }
        extrapolate(velocity, defined, boundary);
    }
};

int main(int argc, char* argv[])
{
    openvdb::initialize();
    openvdb::math::Transform::Ptr trans = openvdb::math::Transform::createLinearTransform(1.0);

    // The container grid which contains the fluid
    openvdb::FloatGrid::Ptr containerGrid = openvdb::FloatGrid::create(-1);
    containerGrid->setTransform(trans);
    containerGrid->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);

    openvdb::FloatGrid::Ptr solidGrid = openvdb::FloatGrid::create();
    solidGrid->setTransform(trans);
    solidGrid->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0);

    // The fluid grid represents the fluid
    openvdb::FloatGrid::Ptr fluidGrid = openvdb::FloatGrid::create(-1);
    fluidGrid->setTransform(trans);
    fluidGrid->fill(openvdb::CoordBBox(openvdb::Coord(-5), openvdb::Coord(5)), 0, true);

    // Adiag represents the diagonal components of the sparse matrix, i.e., A[(i,j,k),(i,j,k)]
    openvdb::FloatGrid::Ptr Adiag = openvdb::FloatGrid::create(0);
    Adiag->setTransform(trans);

    openvdb::FloatGrid::Ptr Aplusj = openvdb::FloatGrid::create(0);
    Aplusj->setTransform(trans);

    openvdb::FloatGrid::Ptr Aplusi = openvdb::FloatGrid::create(0);
    Aplusi->setTransform(trans);

    openvdb::FloatGrid::Ptr Aplusk= openvdb::FloatGrid::create(0);
    Aplusk->setTransform(trans);

    openvdb::FloatGrid::Ptr diver = openvdb::FloatGrid::create(0);
    diver->setTransform(trans);

    openvdb::FloatGrid::Ptr rhs = openvdb::FloatGrid::create(0);
    rhs->setTransform(trans);

    openvdb::Int32Grid::Ptr indices = openvdb::Int32Grid::create(0);
    indices->setTransform(trans);

    openvdb::Vec3dGrid::Ptr vels = openvdb::Vec3dGrid::create(openvdb::Vec3d(0, 1, 0));
    vels->setTransform(trans);
    vels->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), openvdb::Vec3d(0, 0, 0), true);

    openvdb::BoolGrid::Ptr defined = openvdb::BoolGrid::create();
    defined->setTransform(trans);
    defined->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), false);
    openvdb::BoolGrid::Accessor defaccessor = defined->getAccessor();

    // Voxelize all leafs
    vels->tree().voxelizeActiveTiles();
    fluidGrid->tree().voxelizeActiveTiles();
    containerGrid->tree().voxelizeActiveTiles();
    solidGrid->tree().voxelizeActiveTiles();
    defined->tree().voxelizeActiveTiles();
    std::mersenne_twister_engine<uint32_t, 32, 351, 175, 19, 0xccab8ee7, 11, 0xffffffff, 7, 0x31b6ab00, 15, 0xffe50000, 17, 1812433253> mtRand;

    for (openvdb::FloatGrid::ValueOnIter iter = solidGrid->beginValueOn(); iter; ++iter) {
        openvdb::Coord xyz = iter.getCoord();
        if (abs(xyz.x()) > 13 || abs(xyz.y()) > 13 || abs(xyz.z()) > 13)
        {
            iter.setValue(1);
        }
    }
    openvdb::GridPtrVec grids;
    PointList pos;
    const openvdb::Index64 pointCount = 8000;
    openvdb::tools::UniformPointScatter<PointList, std::mersenne_twister_engine<uint32_t, 32, 351, 175, 19, 0xccab8ee7, 11, 0xffffffff, 7, 0x31b6ab00, 15, 0xffe50000, 17, 1812433253>> scatter(pos, pointCount, mtRand);
    scatter.operator()<openvdb::FloatGrid>(*fluidGrid);
    pos.initialize(15, trans);

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double>> cg;

    openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();
    bool isIndiceVoxelized = false;
    bool areVoxelized = false;
    openvdb::Vec3d gravity(0, -10, 0);
    double dx = 1.0;

for (int i = 0; i < 100; ++i)
{
    double dt = 0.1;
    vels->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), openvdb::Vec3d(0, 0, 0), true);
    // grids.push_back(vels -> deepCopy());
    pos.P2Gtransfer(vels);
    // grids.push_back(vels -> deepCopy());
    for (openvdb::Vec3dGrid::ValueOnIter iter = vels->beginValueOn(); iter; ++iter) {
        openvdb::Vec3d xyz = iter.getCoord().asVec3d();
        double val = (dx/(2.0 * (iter.getValue() + dt * gravity).length()));
        dt = val > dt?dt:val;
    }
    std::cout << "Time delta:\t" << dt << std::endl;
    for (openvdb::Vec3dGrid::ValueOnIter iter = vels->beginValueOn(); iter; ++iter) {
        openvdb::Vec3d xyz = iter.getCoord().asVec3d();
        iter.setValue(iter.getValue() + dt * gravity);
    }

    indices->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), -1, true);
    if (!isIndiceVoxelized)
    {
        indices->tree().voxelizeActiveTiles();
        isIndiceVoxelized = true;
    }

    int numActive = 0;
    for (openvdb::FloatGrid::ValueOnIter iter = containerGrid->beginValueOn(); iter; ++iter) {
        openvdb::Vec3d xyz = iter.getCoord().asVec3d();
        double val = pos.interpolate(xyz);
        iter.setValue(val);
        if (val > 0 && !isSolid(iter.getCoord(), solidGrid))
        {
            indaccessor.setValue(iter.getCoord(), numActive);
            numActive++;
        }
    }

    Eigen::VectorXd p(numActive);
    grids.push_back(containerGrid -> deepCopy());
    // grids.push_back(vels -> deepCopy());
    Eigen::VectorXd b(numActive);
    Eigen::VectorXd b2(numActive);
    double error;
    do
    {

        Eigen::SparseMatrix<double> A(numActive, numActive);
        Adiag->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);
        Aplusj->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);
        Aplusi->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);
        Aplusk->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);
        diver->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);
        rhs->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);

        if (!areVoxelized)
        {
            Adiag->tree().voxelizeActiveTiles();
            Aplusi->tree().voxelizeActiveTiles();
            Aplusj->tree().voxelizeActiveTiles();
            Aplusk->tree().voxelizeActiveTiles();
            diver->tree().voxelizeActiveTiles();
            rhs->tree().voxelizeActiveTiles();
            areVoxelized = true;
        }

        setRHS(dx, containerGrid, vels, rhs, solidGrid);
        setDiver(dx, vels, diver, rhs, containerGrid, solidGrid);
        setA(dx, /*dt*/dt, 1, containerGrid, Adiag, Aplusi, Aplusj, Aplusk, solidGrid);
        setA2(indices, b, A, diver, Adiag, Aplusi, Aplusj, Aplusk);
        cg.compute(A);
        p = cg.solve(b);
        velUpdate(indices, dx, /*dt*/dt/10, 1, vels, containerGrid, p, solidGrid);

        diver->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);
        rhs->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);
        setRHS(dx, containerGrid, vels, rhs, solidGrid);
        setDiver(dx, vels, diver, rhs, containerGrid, solidGrid);
        setOnlyB(indices, b2, diver, Adiag);
        // setA2(indices, b2, A, diver, Adiag, Aplusi, Aplusj, Aplusk);
        error = ((b - b2).norm())/(b.norm());
    }while(error > 0.1);
    // openvdb::tools::PointAdvect<openvdb::Vec3dGrid, std::vector<openvdb::Vec3d>, true, openvdb::util::NullInterrupter> pa(*vels);
    // pa.advect(pos.list, dt);
    pos.advect(dt, vels, 15);
    std::cout << "Error:\t" << error << std::endl;
    std::cout << "Iteration:\t" << i+1 << std::endl;
}
    for (openvdb::FloatGrid::ValueOnIter iter = containerGrid->beginValueOn(); iter; ++iter) {
        openvdb::Vec3d xyz = iter.getCoord().asVec3d();
        iter.setValue(pos.interpolate(xyz));
    }

    openvdb::io::File file("mygrids.vdb");
    grids.push_back(containerGrid);
    // grids.push_back(vels);
    file.write(grids);
    file.close();

}
