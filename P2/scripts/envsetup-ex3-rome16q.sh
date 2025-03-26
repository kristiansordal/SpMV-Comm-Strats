#!/bin/bash
# acg: adaptive conjugate gradient algorithms
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Copyright (C) 2024 Simula Research Laboratory
#
# authors: James D. Trotter <james@simula.no>
#
# This script sets up the environment for compiling and running on the
# rome16q partition of the eX3 cluster without CUDA support.
#
# Example usage:
#
#  $ . envsetup-ex3-rome16q.sh
#
#
# ChangeLog:
#
#  2024-08-13: initial version
#

program_name=envsetup-ex3-rome16q.sh
program_version=v1

function help()
{
    printf "Usage: ${0} [OPTION]..\n"
    printf " set up environment on the eX3 cluster's rome16q partition\n"
    printf "\n"
    printf " Options are:\n"
    printf "  %-20s\t%s\n" "--prefix=PREFIX" "install files to PREFIX [/global/D1/homes/james/acg/ex3/rome16q-${program_version}]"
    printf "  %-20s\t%s\n" "-f, --force" "force reloading environment"
    printf "  %-20s\t%s\n" "-v, --verbose" "be more verbose"
    printf "  %-20s\t%s\n" "-h, --help" "display this help and exit"
    printf "  %-20s\t%s\n" "--version" "display version information and exit"
    printf "\n"
    printf " Any additional options are ignored.\n"
    printf "\n"
    printf " Report bugs to: <james@simula.no>.\n"
}

function version()
{
    printf "%s %s\n" "${program_name}" "${program_version}"
    printf "Copyright (C) 2024 Simula Research Laboratory\n"
    printf "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>\n"
    printf "This is free software: you are free to change and redistribute it.\n"
    printf "There is NO WARRANTY, to the extent permitted by law.\n"
}

function parse_program_options()
{
    PREFIX="/global/D1/homes/james/acg/ex3/rome16q-${program_version}"
    FORCE=
    VERBOSE=
    while [ "$#" -gt 0 ]; do
        case "${1}" in
            -h | --help) help; return 1;;
            --version) version; return 1;;
            --prefix) export PREFIX="${2}"; shift 2;;
            --prefix=*) export PREFIX="${1#*=}"; shift 1;;
            -f | --force) FORCE=1; shift 1;;
            -v | --verbose) VERBOSE=1; shift 1;;
            --) shift; break;;
            *) shift; continue;;
        esac
    done

    # add trailing slash, if needed
    [ ${#PREFIX} -gt 0 ] && [ "${PREFIX:${#PREFIX}-1:1}" != "/" ] && PREFIX="${PREFIX}/"
    return 0
}

function main ()
{
    if ! parse_program_options "$@"; then
        return 1
    fi

    if [ ! -z "${ENVSETUP}" ] && [ -z "${FORCE}" ]; then
        echo "${program_name}: environment already loaded - ${ENVSETUP}" >&2
        return 1
    fi

    # create directories
    if [ ! -d "${PREFIX}jobs" ]; then
        [ ! -z "${VERBOSE}" ] && echo "creating directory ${PREFIX}jobs"
        mkdir -p "${PREFIX}jobs"
    fi

    # load modules
    module use /cm/shared/ex3-modules/202309a/defq/modulefiles
    module load cmake-3.22.3
    module load openmpi-4.1.4
    module load metis-32-5.1.0
    module load valgrind-3.19.0
    module load gdb-13.2
    module load sparsebase-0.3.1

    # configure PATH, C_INCLUDE_PATH, LD_LIBRARY_PATH, etc.
    [ ! -z "${VERBOSE}" ] && echo "prepending ${PREFIX}bin to PATH" >&2
    export PATH="${PREFIX}bin:${PATH}"
    [ ! -z "${VERBOSE}" ] && echo "prepending ${PREFIX}lib to LIBRARY_PATH" >&2
    export LIBRARY_PATH="${PREFIX}lib:${LIBRARY_PATH}"
    [ ! -z "${VERBOSE}" ] && echo "prepending ${PREFIX}lib64 to LIBRARY_PATH" >&2
    export LIBRARY_PATH="${PREFIX}lib64:${LIBRARY_PATH}"
    [ ! -z "${VERBOSE}" ] && echo "prepending ${PREFIX}lib to LD_LIBRARY_PATH" >&2
    export LD_LIBRARY_PATH="${PREFIX}lib:${LD_LIBRARY_PATH}"
    [ ! -z "${VERBOSE}" ] && echo "prepending ${PREFIX}lib64 to LD_LIBRARY_PATH" >&2
    export LD_LIBRARY_PATH="${PREFIX}lib64:${LD_LIBRARY_PATH}"
    [ ! -z "${VERBOSE}" ] && echo "prepending ${PREFIX}include to C_INCLUDE_PATH" >&2
    export C_INCLUDE_PATH="${PREFIX}include:${C_INCLUDE_PATH}"
    [ ! -z "${VERBOSE}" ] && echo "prepending ${PREFIX}include to CPLUS_INCLUDE_PATH" >&2
    export C_INCLUDE_PATH="${PREFIX}include:${CPLUS_INCLUDE_PATH}"
    [ ! -z "${VERBOSE}" ] && echo "prepending ${PREFIX}lib/pkgconfig to PKG_CONFIG_PATH" >&2
    export PKG_CONFIG_PATH="${PREFIX}lib/pkgconfig:${PKG_CONFIG_PATH}"
    [ ! -z "${VERBOSE}" ] && echo "prepending ${PREFIX}lib64/pkgconfig to PKG_CONFIG_PATH" >&2
    export PKG_CONFIG_PATH="${PREFIX}lib64/pkgconfig:${PKG_CONFIG_PATH}"
    [ ! -z "${VERBOSE}" ] && echo "prepending ${PREFIX}share/pkgconfig to PKG_CONFIG_PATH" >&2
    export PKG_CONFIG_PATH="${PREFIX}share/pkgconfig:${PKG_CONFIG_PATH}"
    [ ! -z "${VERBOSE}" ] && echo "prepending ${PREFIX}lib/cmake to CMAKE_PREFIX_PATH" >&2
    export CMAKE_PREFIX_PATH="${PREFIX}lib/cmake:${CMAKE_PREFIX_PATH}"
    # [ ! -z "${VERBOSE}" ] && echo "prepending ${PREFIX}lib/python3.7/site-packages to PYTHONPATH" >&2
    # export PYTHONPATH="${PREFIX}lib/python3.7/site-packages:${PYTHONPATH}"

    export ENVSETUP="${program_name} ${program_version} (loaded on $(date))"
}

main "$@"
