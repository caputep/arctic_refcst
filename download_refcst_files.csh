#!/bin/csh

  set yyyymmdd_case  = ${1}

#  source /linuxapps/anaconda3/anacondaenv

  set FHRMAX   = 120
  set FCST_DIR = /rt11/torn/refcst/cyclone_init

  set fhr = 0
  while ( $fhr <= $FHRMAX )

     set yyyymmdd = `echo ${yyyymmdd_case}00 -${fhr}h | ./advance_time | cut -b1-8`

     if ( ! -e ${FCST_DIR}/${yyyymmdd}00_fcst.grb2 ) then

        foreach mem ( c00 p01 p02 p03 p04 p05 p06 p07 p08 p09 p10 )

           set gribfileh = hgt_pres_${yyyymmdd}00_${mem}.grib2
           ./refcst2_cdc.csh $gribfileh ${yyyymmdd}00 $mem

           set gribfileu = ugrd_pres_${yyyymmdd}00_${mem}.grib2
           ./refcst2_cdc.csh $gribfileu ${yyyymmdd}00 $mem

           set gribfilev = vgrd_pres_${yyyymmdd}00_${mem}.grib2
           ./refcst2_cdc.csh $gribfilev ${yyyymmdd}00 $mem

           set gribfilem = pres_msl_${yyyymmdd}00_${mem}.grib2
           ./refcst2_cdc.csh $gribfilem ${yyyymmdd}00 $mem

           set fhrf = 0
           while ( $fhrf <= 120 )

              if ( $fhrf > 0 ) then
                 set fstring = "${fhrf} hour fcst"
              else
                 set fstring = "anl"
              endif

              foreach pres ( 850 925 )
                 wgrib2 -s $gribfileh | grep ":HGT:${pres} mb:"  | grep ":${fstring}:" | wgrib2 -fix_ncep -i -append $gribfileh -grib reforecast.grb2 >! /dev/null
                 wgrib2 -s $gribfileu | grep ":UGRD:${pres} mb:" | grep ":${fstring}:" | wgrib2 -fix_ncep -i -append $gribfileu -grib reforecast.grb2 >! /dev/null
                 wgrib2 -s $gribfilev | grep ":VGRD:${pres} mb:" | grep ":${fstring}:" | wgrib2 -fix_ncep -i -append $gribfilev -grib reforecast.grb2 >! /dev/null
              end

              wgrib2 -s $gribfilem | grep ":${fstring}:" | wgrib2 -fix_ncep -i -append $gribfilem -grib reforecast.grb2 >! /dev/null

              @ fhrf += 6

           end

           rm -rf $gribfileh $gribfileu $gribfilev $gribfilem

        end

        mv -fv reforecast.grb2 ${FCST_DIR}/${yyyymmdd}00_fcst.grb2

     endif

     ln -sf ${FCST_DIR}/${yyyymmdd}00_fcst.grb2 reforecast.grb2
     ncl_convert2nc reforecast.grb2

#     python track_ens_ryan.py -d tr_198501 -i 25028 -l 0

     rm -rf reforecast.nc reforecast.grb2

     @ fhr += 24

  end
