## analysis between two cultures 

data.jp <- read.csv("MergeJP_all_data.csv")
data.th <- read.csv("MergeTh_all_data.csv")

# change column names
colnames(data.jp) <- c("Rating","Animation","HRV_LFHF_mean",
                           "HRV_LFHF_std","id","Avatar")
colnames(data.th) <- c("Rating","Animation","HRV_LFHF_mean",
                       "HRV_LFHF_std","id","Avatar")

data.jp$id <- paste("jp",data.jp$id,sep = "-")
data.th$id <- paste("th",data.th$id,sep = "-")
data.jp$country <- "JP"
data.th$country <- "TH"

#combine data
data.all <- rbind(data.jp,data.th)
data.all$Animation[data.all$Animation == "LetfHandToBody"] <- "LeftHandToBody"

#anova
anova_model <- aov_ez(id = "id", dv = "Rating", 
                      within = c("Animation", "Avatar"),
                      between = c("country"),
                      data = data.all)

# Display the results
summary(anova_model)
emmeans_model <- emmeans(anova_model, pairwise ~ country, adjust = "tukey")
print(emmeans_model$contrasts)

t.test(data.all$Rating[data.all$country == "JP"],data.all$Rating[data.all$country == "TH"])
wilcox.test(data.all$Rating[data.all$country == "JP"],data.all$Rating[data.all$country == "TH"])

# check normality in data (violate normality)
shapiro.test(data.all$Rating[data.all$country == "JP"])

# homogeneity of variance test (violated!!)
library(car)
levene_test <- leveneTest(Rating ~ country, data = data.all)
print(levene_test)
boxplot(Rating ~ country, data = data.all, main = "Boxplot by Group", xlab = "Group", ylab = "Values")

# group data and conduct wilcox.test
summarized_data <- data.all %>% group_by(id) %>% 
  summarise(mean.rating = mean(Rating)) %>% 
  separate(id, into = c("country","number"),sep= '-',remove = FALSE)
wilcox.test(summarized_data$mean.rating[summarized_data$country == "jp"],summarized_data$mean.rating[summarized_data$country == "th"])

# เนื่องจากมี data ซ้ำกันเยอะทำให้เป็น non-parametric
table(summarized_data$mean.rating)

# Within-subjects (Friedman test by group)
data.all %>%
  group_by(country,Animation) %>%
  summarise(Friedman_p_value = friedman.test(Rating ~ Avatar | id)$p.value)

data.all %>%
  group_by(country,Avatar) %>%
  summarise(Friedman_p_value = friedman.test(Rating ~ Animation | id)$p.value)

## plot between TH and JP
# ggplot
library(ggplot2)
summarized_data <- data.all %>% group_by(country) %>% 
  summarise(mean.rating = mean(Rating),
            se = sd(Rating)/sqrt(n()))

# Create bar plot with error bars
ggplot(summarized_data, aes(x = country, y = mean.rating)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "black") +
  geom_errorbar(aes(ymin = mean.rating - se, ymax = mean.rating + se), width = 0.2) +
  labs(title = "Bar Plot with Error Bars", x = "country", y = "Mean Value") +
  theme_minimal()
