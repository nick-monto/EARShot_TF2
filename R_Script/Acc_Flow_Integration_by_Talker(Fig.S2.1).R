data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE),
      se = sd(x[[col]], na.rm=TRUE) / sqrt(length(x[[col]])))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}

acc_Plot <- function(acc_Data, identifier)
{
  summary <- data_summary(acc_Data, 'Accuracy', c('Epoch', 'Pattern_Type'))
  summary$Pattern_Type <- factor(
    summary$Pattern_Type,
    levels= c('Training', 'Pattern_Excluded', 'Identifier_Excluded', 'Test_Only'),
    labels= c('Trained', 'Excluded pattern', 'Excluded identifier', 'Test only')
  )
  
  plot <- ggplot(data= summary, aes(x=Epoch, y=Accuracy, color=Pattern_Type, shape=Pattern_Type)) +
    geom_point(size=3) +
    geom_line(data= subset(summary, Epoch <= epoch_with_Exclusion), aes(x=Epoch, y=Accuracy, color=Pattern_Type, shape=Pattern_Type)) +
    geom_line(data= subset(summary, Epoch > epoch_with_Exclusion), aes(x=Epoch, y=Accuracy, color=Pattern_Type, shape=Pattern_Type)) +
    #geom_text(data= summary, aes(x=Epoch, y=Accuracy + 0.05, label=round(Accuracy, 3))) +
    geom_errorbar(aes(ymin=Accuracy-se, ymax=Accuracy+se), width= max(summary$Epoch) * 0.15, position=position_dodge(0.05)) +
    labs(title= identifier, x="Epoch", y= "Accuracy", colour='Pattern type', shape='Pattern type') +
    ylim(0, 1.1) +
    theme_bw() +
    theme(
      axis.title.x = element_text(size=20),
      axis.title.y = element_text(size=20),
      axis.text.x = element_text(size=13),
      axis.text.y = element_text(size=13),
      panel.grid=element_blank(),
      legend.title = element_text(size=1),
      legend.text = element_text(size=6),
      legend.position = 'bottom',
      plot.title = element_text(hjust = 0.5)
    )
  
  return(plot)
}

library(readr)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(grid)

base_Dir <- 'D:/Python_Programming/EARShot_TF2/Results'
identifier_List <- c('AGNES')
epoch_with_Exclusion <- 4000
epoch_without_Exclusion <- 6000
max_Display_Step <- 60
index <- 0

acc_List <- list()
for (identifier in identifier_List)
{
  work_Dir <- file.path(base_Dir, paste(identifier, '.', 'IDX', index, sep=''), 'Test')
  rt_Data <- read_delim(
    file.path(work_Dir, 'RTs.txt'),
    delim= '\t',
    escape_double= FALSE,
    locale= locale(encoding= 'UTF-8'),
    trim_ws= TRUE
  )
  
  rt_Data <- rt_Data[c(1,3,4,13)]
  rt_Data$Accuracy <- as.numeric(!is.nan(rt_Data$Onset_Time_Dependent_RT))
  
  acc_List[[length(acc_List) + 1]] <- rt_Data
}

acc_Data <- do.call(rbind, acc_List)

acc_Plot_List <- list()
for (identifier in identifier_List)
{
  acc_Data.Subset <- subset(acc_Data, toupper(acc_Data$Identifier) == toupper(identifier))
  plot <- acc_Plot(acc_Data.Subset, identifier)
  acc_Plot_List[[length(acc_Plot_List) + 1]] <- plot
}


png(file.path(base_Dir, sprintf('Accuracy_Flow.IDX_%s.Integration_by_Identifier.png', index)), width = 40, height = 22, res =300, units = "cm")
grid.arrange(arrangeGrob(grobs = acc_Plot_List, ncol=5))
dev.off()