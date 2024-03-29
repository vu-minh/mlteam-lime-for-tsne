sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'

sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo apt update

sudo apt install r-base r-base-core r-recommended r-base-dev

R -e 'install.packages(c("glmnet", "pbapply", "GenSA", "exactRankTests"))'

pip install scipy numpy matplotlib scikit-learn pandas scikit-optimize

mkdir -p ./var/country