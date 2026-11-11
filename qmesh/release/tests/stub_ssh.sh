#!/bin/bash
# stub ssh: consume options (with their -o values), drop the host, run the remote command locally
while [ $# -gt 0 ]; do
  case "$1" in
    -o) shift 2 ;;
    -*) shift ;;
    *)  break ;;
  esac
done
shift   # host
exec bash -c "$*"
