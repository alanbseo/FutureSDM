# join two plant metadat tables
library(dplyr)
library(sf)
library(openxlsx)

data_dir <- file.path(
    "../Data/NIE Data",
    "생태계서비스팀 공간정보",
    "07. 전국자연환경조사"
)

plant_meta1 <- read.xlsx(file.path(
    data_dir,
    "식물종속과목록 _2023083 1(취약300)_수정.xlsx"
))



plant_meta2 <- read.xlsx(file.path(data_dir, "식물종목록(n_code).xlsx"))
print(table(plant_meta2$no == plant_meta2$종코드))


plant_meta_joined <- left_join(plant_meta1, plant_meta2,
    # by = c("종코드" = "no"),
    by = c("종코드" = "종코드"),
    suffix = c("", "_meta2")
)

# here see if all records are attained
tb1 <- table(is.na(plant_meta_joined$nCode))
which(is.na(plant_meta_joined$nCode))

plant_meta_joined[1538, ]


which(plant_meta2$종명 == "Eleutherococcus")
which(plant_meta2$종명 == "물억새")
plant_meta_joined[889, ]
plant_meta2[603, ]
plant_meta2[1288, ]


print(tb1)
stopifnot(length(unique(plant_meta_joined$nCode)) == nrow(plant_meta_joined))
stopifnot(length(unique(plant_meta_joined$종코드)) == nrow(plant_meta_joined))


# check how many records are in plant_meta1 and plant_meta2
print(nrow(plant_meta1))
print(nrow(plant_meta2))
print(nrow(plant_meta_joined))

str(plant_meta_joined)
print(table(plant_meta_joined$과코드 == plant_meta_joined$과코드_meta2))
print(table(plant_meta_joined$속코드 == plant_meta_joined$속코드_meta2))
print(table(plant_meta_joined$과명 == plant_meta_joined$과명_meta2))
print(table(plant_meta_joined$no == plant_meta_joined$종코드))


table(plant_meta_joined$기후변화.취약식물)
table(plant_meta_joined$북방)
table(plant_meta_joined$남방)
table(plant_meta_joined$특산)


plant_meta_joined <- plant_meta_joined %>%
    select(-과명_meta2, -속명_meta2, -종명_meta2, -과코드_meta2, -속코드_meta2, -북방, -남방, -특산)

table(plant_meta_joined$기후변화지표종)

plant_meta_joined$기후변화지표종 <- factor(plant_meta_joined$기후변화지표종, levels = c(1, 2))

table(plant_meta_joined$기후변화지표종)



write.csv(plant_meta_joined,
    file.path(data_dir, "target_plant_species_5Feb2026.csv"),
    row.names = FALSE
)
