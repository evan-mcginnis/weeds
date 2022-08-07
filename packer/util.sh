install_if_missing () {
  for package in "$*"
  do
    dpkg-query -l $package
    if [ $? -gt 0 ]; then
      apt install $package -y
    else
      echo "$package already installed"
    fi
    if [ $? -gt 0 ]; then
      echo "$package install failed"
    fi
  done
  return $?
}

version_installed () {
  version=`pip list | grep $0 | awk '{print $2}'`
  return $version
}
