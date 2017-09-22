rm exec > /dev/null
rm mygrids.vdb > /dev/null
g++ -std=c++11 -pthread -O3 -DNDEBUG -I . -I .. -isystem /home/aakash/packages/boost_1_64_0 -isystem /usr/local/include -isystem /home/aakash/packages/tbb2017_20170604oss/include -o exec $1 -I . -Wl,-rpath,/home/aakash/packages/well/openvdb/build/lib -L/home/aakash/packages/well/openvdb ./openvdb/libopenvdb.so.4.0.2 -ldl -lm -lz -Wl,-rpath,/usr/local/lib -L/usr/local/lib -lHalf -Wl,-rpath,/home/aakash/packages/tbb2017_20170604oss/lib -L/home/aakash/packages/tbb2017_20170604oss/lib -ltbb -Wl,-rpath,/home/aakash/packages/boost_1_64_0/stage/lib -L/home/aakash/packages/boost_1_64_0/stage/lib -lboost_iostreams -lboost_system  -lrt
echo "Compiled"
./exec
./openvdb/build/bin/vdb_view mygrids.vdb