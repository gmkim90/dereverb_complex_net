#!/bin/bash

stage=1


if [ $# -ne 3 ];then
    echo "$0 <mc_wsj_root> <wsjcam0_root> <dst>"
    echo "e.g."
    echo "$0 /export/corpora/LDC2014S03 /export/corpora/LDC95S24 REVERB_DATA_OFFICIAL"
    exit 1
fi

set -euo pipefail
which matlab &> /dev/null || { echo "$0: Error: requires matlab"; exit 1; }
which sox &> /dev/null || { echo "$0: Error: requires sox"; exit 1; }

mc_wsj_root=$1
wsjcam0_root=$2
dir=$3

if [ "${stage}" -le 1 ]; then
    echo "$0: 1. Split 'LDC2014S03: Multi-Channel WSJ Audio' into Dev and Eval"

    # LDC2014S03: Multi-Channel WSJ Audio

    # 1. Validates the directory structure
    # LDC2014S03/
    #         |- multi-ch-wsj_aud/
    #                         |- data/
    #                              |- audio/
    #                                   |- move/ # Not used
    #                                   |- olap/ # Not used
    #                                   |- stat/
    #                              |- etc/
    #                              |- mlf/
    if [ ! -e "${mc_wsj_root}/multi-ch-wsj_aud" ]; then
        echo "Error: not found: ${mc_wsj_root}/multi-ch-wsj_aud"
        exit 1
    fi
    if [ ! -e "${mc_wsj_root}/multi-ch-wsj_aud/data" ]; then
        echo "Error: not found: ${mc_wsj_root}/multi-ch-wsj_aud/data"
        exit 1
    fi
    for name in audio etc mlf; do
        if [ ! -e "${mc_wsj_root}/multi-ch-wsj_aud/data/${name}" ]; then
            echo "Error: not found: ${mc_wsj_root}/multi-ch-wsj_aud/data/${name}"
            exit 1
        fi
    done

    # 2. Creates REVERB_DATA_OFFICIAL/MC_WSJ_AV_{Dev|Eval}
    # REVERB_DATA_OFFICIAL/
    #                 |- MC_WSJ_AV_Dev/
    #                              |- audio/
    #                                   |- stat/
    #                              |- etc/
    #                              |- mlf/
    #                 |- MC_WSJ_AV_Eval/
    #                              |- audio/
    #                                   |- stat/
    #                              |- etc/
    #                              |- mlf/
    for name in MC_WSJ_AV_Dev MC_WSJ_AV_Eval; do
        mkdir -p "${dir}/${name}"
        mkdir -p "${dir}/${name}/audio/stat"
        for name2 in etc mlf; do
            rsync -arz "${mc_wsj_root}/multi-ch-wsj_aud/data/${name2}" "${dir}/${name}/"
        done
    done

    for name3 in T6 T7 T8 T9 T10; do
        for w in $(find ${mc_wsj_root}/multi-ch-wsj_aud/data/audio/stat/${name3} -name '*.flac'); do
            basename="${w#${mc_wsj_root}/multi-ch-wsj_aud/data/}"
            basename="${basename%.flac}"
            outwav="${dir}/MC_WSJ_AV_Dev/${basename}.wav"
            if [ ! -e "${outwav}" ]; then
                mkdir -p $(dirname ${outwav})
                sox ${w} ${outwav}
            fi
        done
    done

    for name3 in T21 T22 T23 T24 T25 T36 T37 T38 T39 T40; do
        for w in $(find ${mc_wsj_root}/multi-ch-wsj_aud/data/audio/stat/${name3} -name '*.flac'); do
            basename="${w#${mc_wsj_root}/multi-ch-wsj_aud/data/}"
            basename="${basename%.flac}"
            outwav="${dir}/MC_WSJ_AV_Eval/${basename}.wav"
            if [ ! -e "${outwav}" ]; then
                mkdir -p $(dirname ${outwav})
                sox ${w} ${outwav}
            fi
        done
    done
fi


if [ "${stage}" -le 2 ]; then
    echo "$0: 2. Generating SimData from 'LDC95S24: WSJCAM0 Cambridge Read News'"

    mkdir -p ${dir}/downloads
    # Get voicebox
    if [ ! -e ${dir}/downloads/sap-voicebox ]; then
        git clone https://github.com/ImperialCollegeLondon/sap-voicebox.git ${dir}/downloads/sap-voicebox
    fi
    export MATLABPATH=$(cd ${dir}; pwd)/downloads/sap-voicebox/voicebox

    for f in reverb_tools_for_Generate_mcTrainData reverb_tools_for_Generate_SimData; do
        if [ ! -e ${dir}/${f} ]; then
            if [ ! -e ${dir}/downloads/${f}.tgz ]; then
                wget https://reverb2014.dereverberation.com/tools/${f}.tgz -O ${dir}/downloads/${f}.tgz
            fi
            tar zxvf ${dir}/downloads/${f}.tgz -C ${dir}
        fi
    done

    if ! which sph2pipe &> /dev/null; then
        if [ ! -e ${dir}/downloads/sph2pipe_v2.5.tar.gz ]; then
            wget https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz -O ${dir}/downloads/sph2pipe_v2.5.tar.gz
        fi
        if [ ! -e ${dir}/downloads/sph2pipe_v2.5/sph2pipe ]; then
            tar zxvf ${dir}/downloads/sph2pipe_v2.5.tar.gz -C ${dir}/downloads
            pushd ${dir}/downloads/sph2pipe_v2.5; gcc -o sph2pipe *.c -lm; pushd
        fi
        export PATH=${dir}/downloads/sph2pipe_v2.5:${PATH}
    fi

    dir_abs=$(cd ${dir}; pwd)

    pushd ${dir}/reverb_tools_for_Generate_mcTrainData
    mkdir -p bin
    for f in h_strip w_decode; do
        if [ ! -e bin/${f} ]; then
            rm -f bin/${f}
            echo "cd bin; ln -s ../../reverb_tools_for_Generate_SimData/bin/${f} .; cd .."
            cd bin; ln -s ../../reverb_tools_for_Generate_SimData/bin/${f} .; cd ..
        fi
    done
    chmod u+x sphere_to_wave.csh
    chmod u+x bin/*

    sed -i.bak -e "s#save_dir='.*';#save_dir='${dir_abs}/REVERB_WSJCAM0_tr';#" Generate_mcTrainData.m
    matlab -nodisplay -nosplash -r "Generate_mcTrainData('$wsjcam0_root'); exit"
    pushd


    pushd ${dir}/reverb_tools_for_Generate_SimData
    chmod u+x sphere_to_wave.csh
    chmod u+x bin/*

    matlab -nodisplay -nosplash -r "Generate_dtData('${wsjcam0_root}','${dir_abs}/REVERB_WSJCAM0_dt'); exit"
    matlab -nodisplay -nosplash -r "Generate_etData('${wsjcam0_root}','${dir_abs}/REVERB_WSJCAM0_et'); exit"
    pushd

fi
