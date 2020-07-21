version="3.3.7"
eigen="eigen-$version"
echo "Installing Dependenices"
sudo apt-get -y install build-essential checkinstall cmake pkg-config yasm
echo "Downloading $eigen"
mkdir $eigen
cd $eigen
wget -O $eigen.tar.bz2 https://gitlab.com/libeigen/eigen/-/archive/$version/eigen-$version.tar.bz2
mkdir $eigen
tar --strip-components=1 -xvjf $eigen.tar.bz2 -C $eigen
echo "Installing $eigen"
cd $eigen
mkdir build
cd build
cmake ..
make
sudo make install
echo "Finished: $eigen is ready to be used"
