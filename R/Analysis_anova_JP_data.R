library(afex)
library(emmeans)
library(dplyr)
library(tidyr)

# anova for data rating
data.rating <- read.csv("rating-data.csv")
data.q1 <- read.csv("rating_q1.csv") %>% 
  mutate(Avatar = tolower(Avatar))
data.q1rating <- data.q1 %>% select(-HRV_LFHF_mean)
data.rating <- rbind(data.rating,data.q1rating)

## mean , sd of each avatar
data.rating %>% 
  group_by(id,Avatar) %>% 
  summarise(Rating = mean(Rating)) %>% 
  group_by(Avatar) %>% 
  summarise(mean_Rating = mean(Rating),
            sd_Rating = sd(Rating))

## mean , sd of each animation
data.rating %>% 
  group_by(id,Animation) %>% 
  summarise(Rating = mean(Rating)) %>% 
  group_by(Animation) %>% 
  summarise(mean_Rating = mean(Rating),
            sd_Rating = sd(Rating))


anova_model <- aov_ez(id = "id", dv = "Rating", 
                      within = c("Animation", "Avatar"), 
                      data = data.rating)

# Display the results
summary(anova_model)
emmeans_model <- emmeans(anova_model, pairwise ~ Avatar, adjust = "tukey")
print(emmeans_model$contrasts)

emmeans_model <- emmeans(anova_model, pairwise ~ Animation, adjust = "tukey")
print(emmeans_model$contrasts)


# anova for LFHF data
data.lfhf <- read.csv("LFHF-data.csv")
colnames(data.lfhf)[4] <- "HRV_LFHF_mean"
data.q1lfhf <- data.q1 %>% select(-Rating)
data.lfhf <- rbind(data.lfhf,data.q1lfhf)

## mean , sd of each avatar
data.lfhf %>% 
  group_by(id,Avatar) %>% 
  summarise(mean_HRV = mean(HRV_LFHF_mean)) %>% 
  group_by(Avatar) %>% 
  summarise(HRV_LFHF_mean = mean(mean_HRV),
            sd_LFHF = sd(mean_HRV))

## mean , sd of each animation
data.rating %>% 
  group_by(id,Animation) %>% 
  summarise(Rating = mean(Rating)) %>% 
  group_by(Animation) %>% 
  summarise(mean_Rating = mean(Rating),
            sd_Rating = sd(Rating))

anova_model <- aov_ez(id = "id", dv = "HRV_LFHF_mean", 
                      within = c("Animation", "Avatar"), 
                      data = data.lfhf)

# Display the results
summary(anova_model)

##change column names
colnames(data.rating)[4] <- "Rating"
colnames(data.rating)
colnames(data.lfhf)[4] <- "LFHF"
colnames(data.lfhf)

# fix mispelling animation name 
data.rating %>% distinct(Animation)
data.lfhf %>% distinct(Animation)
data.lfhf$Animation[data.lfhf$Animation == "LetfHandToBody"] <- "LeftHandToBody"

# merge data
merged_data <- merge(data.rating, data.lfhf, by = c("Avatar", "id", "Animation"))

plot(merged_data[,"Rating"],merged_data[,"LFHF"])

merged_data %>% 
  group_by(id) %>% 
  summarise(mean_Rating_rating = mean(Rating),
            mean_LFHF = mean(LFHF)) -> gdata
  
# find correlation
gdata <- as.data.frame(gdata)
plot(gdata[,"mean_Rating_rating"],gdata[,"mean_LFHF"])
cor.test(gdata[,"mean_Rating_rating"],gdata[,"mean_LFHF"])

# find correlation
merged_data %>% 
  group_by(id,Avatar) %>% 
  summarise(mean_Rating_rating = mean(Rating),
            mean_LFHF = mean(LFHF)) -> gdata

gdata <- as.data.frame(gdata)
plot(gdata[,"mean_Rating_rating"],gdata[,"mean_LFHF"])
cor.test(gdata[,"mean_Rating_rating"],gdata[,"mean_LFHF"])

# find correlation
merged_data %>% 
  group_by(id,Animation) %>% 
  summarise(mean_Rating_rating = mean(Rating),
            mean_LFHF = mean(LFHF)) -> gdata

gdata <- as.data.frame(gdata)
plot(gdata[,"mean_Rating_rating"],gdata[,"mean_LFHF"])
cor.test(gdata[,"mean_Rating_rating"],gdata[,"mean_LFHF"])


##### end of correlation
####
merged_data %>% 
  group_by(Avatar) %>% 
  summarise(mean_Rating_rating = mean(Rating_rating),
            mean_LFHF = mean(LFHF)) -> gdata

## rating score
## find new value to compute new error bar
merged_data <- data.rating
merged_data %>% 
  group_by(id,Avatar) %>% 
  summarise(mean_Rating_rating = mean(Rating)) -> gdata

gdata <- as.data.frame(gdata)
grand.mean <- mean(gdata[,"mean_Rating_rating"])

merged_data %>% 
  group_by(id) %>% 
  summarise(mean_Rating_rating = mean(Rating)) -> subject.mean 
subject.mean <- as.data.frame(subject.mean)

gdata %>% 
  left_join(subject.mean,by= "id") -> newvalue

newvalue <- as.data.frame(newvalue)
newvalue["grand.mean"] <- grand.mean

newvalue["new.rating"] <- newvalue[,"mean_Rating_rating.x"] - newvalue[,"mean_Rating_rating.y"] + newvalue[,"grand.mean"]
# Function to compute standard error
error_bar <- function(x) {
  1.96 * (sd(x) / sqrt(length(x)))
}
result <- newvalue %>% group_by(Avatar) %>% 
  summarise(
    mean_avatar = mean(mean_Rating_rating.x),
    errbar_avatar = error_bar(new.rating)
  )
result <- as.data.frame(result)

# ggplot
library(ggplot2)
ggplot(result) +
  geom_bar(aes(x=Avatar, y=mean_avatar), stat='identity') +
  geom_errorbar(aes(x=Avatar, ymin=mean_avatar-errbar_avatar, ymax=mean_avatar+errbar_avatar), width=0.4)


# Assuming 'result' is your data frame with columns 'Avatar', 'mean_avatar', and 'errbar_avatar'
ggplot(result, aes(x = Avatar, y = mean_avatar)) +
  geom_bar(stat = 'identity', fill = "skyblue", color = "black", width = 0.6) +  # Customize bar color and width
  geom_errorbar(aes(ymin = mean_avatar - errbar_avatar, ymax = mean_avatar + errbar_avatar), 
                width = 0.2, color = "black") +  # Customize error bars
  labs(
    title = "Discomfort Ratings by Avatar with Error Bars",
    x = "Avatar", 
    y = "Mean Rating Score"
  ) +  # Add title and axis labels
  scale_y_continuous(limits = c(0, 4), expand = c(0, 0)) +  # Set y-axis range and remove extra space
  scale_x_discrete(labels = function(x) { tools::toTitleCase(tolower(x)) }) +  # Capitalize only the first letter of each label
  theme_minimal() +  # Use a clean theme
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),  # Center and style title
    axis.text = element_text(size = 12),  # Adjust axis text size
    axis.title = element_text(size = 13, face = "bold")  # Adjust axis title size
  )

##----------------------
## LFHF score
## find new value to compute new error bar
merged_data <- data.lfhf
merged_data %>% 
  group_by(id,Avatar) %>% 
  summarise(mean_LFHF = mean(LFHF)) -> gdata

gdata <- as.data.frame(gdata)
grand.mean <- mean(gdata[,"mean_LFHF"])

merged_data %>% 
  group_by(id) %>% 
  summarise(mean_lfhf = mean(LFHF)) -> subject.mean 
subject.mean <- as.data.frame(subject.mean)

gdata %>% 
  left_join(subject.mean,by= "id") -> newvalue

newvalue <- as.data.frame(newvalue)
newvalue["grand.mean"] <- grand.mean

newvalue["new.lfhf"] <- newvalue[,"mean_LFHF"] - newvalue[,"mean_lfhf"] + newvalue[,"grand.mean"]
# Function to compute standard error
error_bar <- function(x) {
  1.96 * (sd(x) / sqrt(length(x)))
}
result <- newvalue %>% group_by(Avatar) %>% 
  summarise(
    mean_avatar = mean(mean_LFHF),
    errbar_avatar = error_bar(new.lfhf)
  )
result <- as.data.frame(result)

# ggplot
library(ggplot2)
ggplot(result) +
  geom_bar(aes(x=Avatar, y=mean_avatar), stat='identity') +
  geom_errorbar(aes(x=Avatar, ymin=mean_avatar-errbar_avatar, ymax=mean_avatar+errbar_avatar), width=0.4)


# Assuming 'result' is your data frame with columns 'Avatar', 'mean_avatar', and 'errbar_avatar'
ggplot(result, aes(x = Avatar, y = mean_avatar)) +
  geom_bar(stat = 'identity', fill = "skyblue", color = "black", width = 0.6) +  # Customize bar color and width
  geom_errorbar(aes(ymin = mean_avatar - errbar_avatar, ymax = mean_avatar + errbar_avatar), 
                width = 0.2, color = "black") +  # Customize error bars
  labs(
    title = "LF/HF Ratio by Avatar with Error Bars",
    x = "Avatar", 
    y = "Mean LF/HF Ratio"
  ) +  # Add title and axis labels
  scale_y_continuous(limits = c(0, 4), expand = c(0, 0)) +  # Set y-axis range and remove extra space
  scale_x_discrete(labels = function(x) { tools::toTitleCase(tolower(x)) }) +  # Capitalize only the first letter of each label
  theme_minimal() +  # Use a clean theme
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),  # Center and style title
    axis.text = element_text(size = 12),  # Adjust axis text size
    axis.title = element_text(size = 13, face = "bold")  # Adjust axis title size
  )

###------------------
## rating score by animation
## find new value to compute new error bar
merged_data <- data.rating

merged_data %>% 
  group_by(id,Animation) %>% 
  summarise(mean_Rating_rating = mean(Rating)) -> gdata

gdata <- as.data.frame(gdata)
grand.mean <- mean(gdata[,"mean_Rating_rating"])

merged_data %>% 
  group_by(id) %>% 
  summarise(mean_Rating_rating = mean(Rating)) -> subject.mean 
subject.mean <- as.data.frame(subject.mean)

gdata %>% 
  left_join(subject.mean,by= "id") -> newvalue

newvalue <- as.data.frame(newvalue)
newvalue["grand.mean"] <- grand.mean

newvalue["new.rating"] <- newvalue[,"mean_Rating_rating.x"] - newvalue[,"mean_Rating_rating.y"] + newvalue[,"grand.mean"]
# Function to compute standard error
error_bar <- function(x) {
  1.96 * (sd(x) / sqrt(length(x)))
}
result <- newvalue %>% group_by(Animation) %>% 
  summarise(
    mean_animation = mean(mean_Rating_rating.x),
    errbar_animation = error_bar(new.rating)
  )
result <- as.data.frame(result)

# ggplot
library(ggplot2)
ggplot(result) +
  geom_bar(aes(x=Animation, y=mean_animation), stat='identity') +
  geom_errorbar(aes(x=Animation, ymin=mean_animation-errbar_animation, ymax=mean_animation+errbar_animation), width=0.4)



# Assuming 'result' is your data frame with columns 'Animation', 'mean_animation', and 'errbar_animation'
ggplot(result, aes(x = Animation, y = mean_animation)) +
  geom_bar(stat = 'identity', fill = "skyblue", color = "black", width = 0.6) +  # Customize bar color and width
  geom_errorbar(aes(ymin = mean_animation - errbar_animation, ymax = mean_animation + errbar_animation), 
                width = 0.3, color = "black", size = 0.7) +  # Customize error bars
  labs(
    title = "Mean Animation Scores with Error Bars",
    x = "Animation Type", 
    y = "Mean Score"
  ) +  # Add title and axis labels
  scale_y_continuous(expand = c(0, 0), limits = c(0, 4)) +  # Remove extra space and set lower limit to 0
  theme_minimal() +  # Use a clean theme
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and style the title
    axis.text = element_text(size = 12),  # Adjust axis text size
    axis.title = element_text(size = 13, face = "bold"),  # Adjust axis title size
    panel.grid.major = element_line(color = "gray90"),  # Customize grid lines
    panel.grid.minor = element_blank()  # Remove minor grid lines
  ) + 
  geom_text(aes(label = round(mean_animation, 2)), vjust = -0.5, size = 4)  # Add text labels above bars

result$Animation[result$Animation == "LeftHandToBody"] <- "LeftHandBOdy"
result$Animation[result$Animation == "LeftHandToFace"] <- "LeftHandFAce"
result$Animation[result$Animation == "RightHandToBody"] <- "RightHandBOdy"
result$Animation[result$Animation == "RightHandToFace"] <- "RightHandFAce"
result$Animation[result$Animation == "TouchBody"] <- "BothHandBOdy"
result$Animation[result$Animation == "TouchFace"] <- "BothHandFAce"
result$Animation <- factor(result$Animation,levels = c("Walk","LeftHandBOdy","RightHandBOdy","BothHandBOdy",
                                                       "LeftHandFAce","RightHandFAce","BothHandFAce"))
# Wrap long labels to fit better
ggplot(result, aes(x = Animation, y = mean_animation)) +
  geom_bar(stat = 'identity', fill = "skyblue", color = "black", width = 0.6) +  
  geom_errorbar(aes(ymin = mean_animation - errbar_animation, ymax = mean_animation + errbar_animation), 
                width = 0.3, color = "black", size = 0.7) +  
  labs(
    title = "Discomfort Ratings by Animation with Error Bars",
    x = "Animation", 
    y = "Mean Rating Score"
  ) +  
  scale_x_discrete(labels = abbreviate) +  # Abbreviate the labels
  scale_y_continuous(expand = c(0, 0), limits = c(0, 4)) +  
  theme_minimal() +  
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  
    axis.text.x = element_text(size = 12),  
    axis.text.y = element_text(size = 12),  
    axis.title = element_text(size = 13, face = "bold")
  )


##----------------------
## LFHF score by animation
## find new value to compute new error bar
merged_data <- data.lfhf
merged_data %>% 
  group_by(id,Animation) %>% 
  summarise(mean_LFHF = mean(LFHF)) -> gdata

gdata <- as.data.frame(gdata)
grand.mean <- mean(gdata[,"mean_LFHF"])

merged_data %>% 
  group_by(id) %>% 
  summarise(mean_lfhf = mean(LFHF)) -> subject.mean 
subject.mean <- as.data.frame(subject.mean)

gdata %>% 
  left_join(subject.mean,by= "id") -> newvalue

newvalue <- as.data.frame(newvalue)
newvalue["grand.mean"] <- grand.mean

newvalue["new.lfhf"] <- newvalue[,"mean_LFHF"] - newvalue[,"mean_lfhf"] + newvalue[,"grand.mean"]
# Function to compute standard error
error_bar <- function(x) {
  1.96 * (sd(x) / sqrt(length(x)))
}
result <- newvalue %>% group_by(Animation) %>% 
  summarise(
    mean_Animation = mean(mean_LFHF),
    errbar_Animation = error_bar(new.lfhf)
  )
result <- as.data.frame(result)

# ggplot
library(ggplot2)
ggplot(result) +
  geom_bar(aes(x=Animation, y=mean_Animation), stat='identity') +
  geom_errorbar(aes(x=Animation, ymin=mean_Animation-errbar_Animation, ymax=mean_Animation+errbar_Animation), width=0.4)


result$Animation[result$Animation == "LeftHandToBody"] <- "LeftHandBOdy"
result$Animation[result$Animation == "LeftHandToFace"] <- "LeftHandFAce"
result$Animation[result$Animation == "RightHandToBody"] <- "RightHandBOdy"
result$Animation[result$Animation == "RightHandToFace"] <- "RightHandFAce"
result$Animation[result$Animation == "TouchBody"] <- "BothHandBOdy"
result$Animation[result$Animation == "TouchFace"] <- "BothHandFAce"
result$Animation <- factor(result$Animation,levels = c("Walk","LeftHandBOdy","RightHandBOdy","BothHandBOdy",
                                                       "LeftHandFAce","RightHandFAce","BothHandFAce"))
# Wrap long labels to fit better
ggplot(result, aes(x = Animation, y = mean_Animation)) +
  geom_bar(stat = 'identity', fill = "skyblue", color = "black", width = 0.6) +  
  geom_errorbar(aes(ymin = mean_Animation - errbar_Animation, ymax = mean_Animation + errbar_Animation), 
                width = 0.3, color = "black", size = 0.7) +  
  labs(
    title = "LF/HF Ratio by Animation with Error Bars",
    x = "Animation", 
    y = "Mean LF/HF Ratio"
  ) +  
  scale_x_discrete(labels = abbreviate) +  # Abbreviate the labels
  scale_y_continuous(expand = c(0, 0), limits = c(0, 4)) +  
  theme_minimal() +  
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  
    axis.text.x = element_text(size = 12),  
    axis.text.y = element_text(size = 12),  
    axis.title = element_text(size = 13, face = "bold")
  )

####################
## recheck data JP
# anova for data rating

data.rating_2 <- read.csv("MergeJP_all_data.csv")

# change column names
colnames(data.rating) <- c("Rating","Animation","HRV_LFHF_mean",
                           "HRV_LFHF_std","id","Avatar")

## mean , sd of each avatar
data.rating %>% 
  group_by(id,Avatar) %>% 
  summarise(Rating = mean(Rating)) %>% 
  group_by(Avatar) %>% 
  summarise(mean_Rating = mean(Rating),
            sd_Rating = sd(Rating))

## mean , sd of each animation
data.rating %>% 
  group_by(id,Animation) %>% 
  summarise(Rating = mean(Rating)) %>% 
  group_by(Animation) %>% 
  summarise(mean_Rating = mean(Rating),
            sd_Rating = sd(Rating))


anova_model <- aov_ez(id = "id", dv = "Rating", 
                      within = c("Animation", "Avatar"), 
                      data = data.rating)

# Display the results
summary(anova_model)
emmeans_model <- emmeans(anova_model, pairwise ~ Avatar, adjust = "tukey")
print(emmeans_model$contrasts)

emmeans_model <- emmeans(anova_model, pairwise ~ Animation, adjust = "tukey")
print(emmeans_model$contrasts)

## LFHF
## mean , sd of each avatar
data.rating %>% 
  group_by(id,Avatar) %>% 
  summarise(mean_HRV = mean(HRV_LFHF_mean)) %>% 
  group_by(Avatar) %>% 
  summarise(HRV_LFHF_mean = mean(mean_HRV),
            sd_LFHF = sd(mean_HRV))

## mean , sd of each animation
data.rating %>% 
  group_by(id,Animation) %>% 
  summarise(mean_HRV = mean(HRV_LFHF_mean)) %>% 
  group_by(Animation) %>% 
  summarise(mean_Rating = mean(mean_HRV),
            sd_Rating = sd(mean_HRV))

anova_model <- aov_ez(id = "id", dv = "HRV_LFHF_mean", 
                      within = c("Animation", "Avatar"), 
                      data = data.rating)

# Display the results
summary(anova_model)


# There is a mistake, so we need to debug it!!
### มีค่า mean ของ rating man and women ไม่ตรงกัน 
data.rating <- read.csv("rating-data_ไม่เอา.csv")
data.q1 <- read.csv("rating_q1.csv") %>% 
  mutate(Avatar = tolower(Avatar))
data.q1rating <- data.q1 %>% select(-HRV_LFHF_mean)

data.rating <- rbind(data.rating,data.q1rating)

data.rating_2 <- read.csv("MergeJP_all_data.csv")

# change column names
colnames(data.rating_2) <- c("Rating_2","Animation","HRV_LFHF_mean",
                           "HRV_LFHF_std","id","Avatar")

data.rating$Animation[data.rating$Animation == "LeftHandToBody"] <- "LetfHandToBody"
data.rating$Animation[data.rating$Animation == "LeftHandToFace"] <- "LeftHandToFace"
data.rating$Animation[data.rating$Animation == "RightHandToBody"] <- "RightHandToBody"
data.rating$Animation[data.rating$Animation == "RightHandToFace"] <- "RightHandToFace"
data.rating$Animation[data.rating$Animation == "TouchBody"] <- "TouchBody"
data.rating$Animation[data.rating$Animation == "TouchFace"] <- "TouchFace"

data.rating$Avatar[data.rating$Avatar == "man"] <- "Man"
data.rating$Avatar[data.rating$Avatar == "woman"] <- "Woman"
data.rating$Avatar[data.rating$Avatar == "robot"] <- "Robot"

left_join_data <- left_join(data.rating,data.rating_2, by = c("id","Avatar","Animation"))
filter_data <- left_join_data[left_join_data$Rating != left_join_data$Rating_2,]
filter_data <- filter_data %>% arrange(id)
write.csv(filter_data,file="not_match_data.csv")
