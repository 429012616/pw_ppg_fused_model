set(srcs
    "main.c"
    "soft_spi.c"
    "soft_spi.S"
    )
set(incs
    "."
)
idf_component_register(SRCS ${srcs}
                    INCLUDE_DIRS ${incs}
                    )

idf_build_set_property(COMPILE_OPTIONS "-Wno-stringop-overflow" APPEND)
#o2 seems not work on i2c_master.c driver,it'll be block forever.
#set_source_files_properties("main.c" PROPERTIES COMPILE_FLAGS "-O1")