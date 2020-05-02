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
    labels= c('Trained', 'Excluded pattern', 'Excluded Identifier', 'Test only')
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

flow_Plot <- function(flow_Data, epoch)
{
  flow_Data$Pattern_Type <- factor(
    flow_Data$Pattern_Type,
    levels= c('Training', 'Pattern_Excluded', 'Identifier_Excluded', 'Test_Only'),
    labels= c('Trained', 'Excluded pattern', 'Excluded identifier', 'Test only')
    )
  flow_Data$Category <- factor(
    flow_Data$Category,
    levels= c('Target', 'Cohort', 'Rhyme', 'Unrelated'),
    labels= c('Target', 'Cohort', 'Rhyme', 'Unrelated')
    )
  
  plot <- ggplot(data= flow_Data, aes(x=Time_Step, y=Cosine_Similarity, color=Category, shape=Category)) +
    geom_line() +
    geom_point(data=subset(flow_Data, Time_Step %% 10 == 0), aes(x=Time_Step, y=Cosine_Similarity, color=Category, shape=Category), size = 3) +
    scale_x_continuous(breaks = seq(0, max_Display_Step, max_Display_Step / 4), labels = seq(0, max_Display_Step, max_Display_Step / 4)*10) +
    facet_grid(.~Pattern_Type) +
    ylim(0,1.1) +
    labs(title= sprintf('CS flow    Epoch: %s', epoch), x='Time step', y='Cosine similarity') +
    theme_bw() +
    theme(
      text = element_text(size=16),
      panel.background = element_blank(),
      panel.grid.major = element_blank(),  #remove major-grid labels
      panel.grid.minor = element_blank(),  #remove minor-grid labels
      plot.background = element_blank(),
      plot.title = element_text(hjust = 0.5)
      )
  
  return(plot)
}

flow_Plot2 <- function(flow_Data, epoch)#No distinguish pattern type
{
  flow_Data$Category <- factor(
    flow_Data$Category,
    levels= c('Target', 'Cohort', 'Rhyme', 'Unrelated'),
    labels= c('Target', 'Cohort', 'Rhyme', 'Unrelated')
  )
  
  plot <- ggplot(data= flow_Data, aes(x=Time_Step, y=Cosine_Similarity, color=Category, shape=Category)) +
    geom_line() +
    geom_point(data=subset(flow_Data, Time_Step %% 10 == 0), aes(x=Time_Step, y=Cosine_Similarity, color=Category, shape=Category), size = 3) +
    scale_x_continuous(breaks = seq(0, max_Display_Step, max_Display_Step / 4), labels = seq(0, max_Display_Step, max_Display_Step / 4)*10) +
    ylim(0,1.1) +
    labs(title= sprintf('CS flow    Epoch: %s', epoch), x='Time step', y='Cosine similarity', colour='', shape='') +
    theme_bw() +
    theme(
      text = element_text(size=12),
      panel.background = element_blank(),
      panel.grid.major = element_blank(),  #remove major-grid labels
      panel.grid.minor = element_blank(),  #remove minor-grid labels
      plot.background = element_blank(),
      legend.position = 'bottom',
      legend.text = element_text(size=7),
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
flow_with_Exclusion_List <- list()
flow_without_Exclusion_List <- list()

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

  rt_Data <- rt_Data[c(1,4,13)]
  rt_Data$Accuracy <- as.numeric(!is.nan(rt_Data$Onset_Time_Dependent_RT))
  rt_Data$Excluded_Identifier <- identifier
  
  acc_List[[length(acc_List) + 1]] <- rt_Data

  
  categorized_Flow_Data <- read_delim(
    file.path(work_Dir, 'Category_Flows.txt'),
    delim= '\t',
    escape_double = FALSE,
    locale = locale(encoding = "UTF-8"),
    trim_ws = TRUE
  )
  for(epoch in c(epoch_with_Exclusion, epoch_without_Exclusion))
  {
    categorized_Flow_Data.Subset <- subset(na.omit(categorized_Flow_Data), Accuracy & Epoch == epoch)[-c(1,2,3,5,6,7,8,9,10,12,13)]
    if (nrow(categorized_Flow_Data) == 0)
    {
      next
    }

    categorized_Flow_Data.Subset.Mean <- aggregate(categorized_Flow_Data.Subset[3:length(categorized_Flow_Data.Subset)], by=list(categorized_Flow_Data.Subset$Pattern_Type, categorized_Flow_Data.Subset$Category), FUN=mean)
    colnames(categorized_Flow_Data.Subset.Mean)[1:2] <- c('Pattern_Type', 'Category')
    categorized_Flow_Data.Subset.Melt <- melt(categorized_Flow_Data.Subset.Mean, id.vars = c('Pattern_Type', 'Category'), variable.name = 'Time_Step', value.name = 'Cosine_Similarity')
    categorized_Flow_Data.Subset.Melt$Time_Step <- as.numeric(categorized_Flow_Data.Subset.Melt$Time_Step)
    categorized_Flow_Data.Subset.Melt <- subset(categorized_Flow_Data.Subset.Melt, Time_Step <= max_Display_Step)
    categorized_Flow_Data.Subset.Melt$Excluded_Identifier <- identifier

    if (epoch == epoch_with_Exclusion)
    {
      flow_with_Exclusion_List[[length(flow_with_Exclusion_List) + 1]] <- categorized_Flow_Data.Subset.Melt
    }
    if (epoch == epoch_without_Exclusion)
    {
      flow_without_Exclusion_List[[length(flow_without_Exclusion_List) + 1]] <- categorized_Flow_Data.Subset.Melt
    }
  }
}

acc_Data <- do.call(rbind, acc_List)
flow_with_Exclusion_Data <- do.call(rbind, flow_with_Exclusion_List)
flow_without_Exclusion_Data <- do.call(rbind, flow_without_Exclusion_List)

acc_Plot_List <- list()
flow_with_Exclusion_Plot_List <- list()
flow_without_Exclusion_Plot_List <- list()

for (identifier in identifier_List)
{
  work_Dir <- file.path(base_Dir, paste(identifier, '.', 'IDX', index, sep=''), 'Test')
  
  acc_Data.Subset <- subset(acc_Data, Excluded_Identifier == identifier)
  plot <- acc_Plot(acc_Data.Subset, identifier)
  ggsave(filename = file.path(work_Dir, "Accuracy_Flow.png"), plot = plot, device = "png", width = 12, height = 12, units = "cm", dpi = 300)
  acc_Plot_List[[length(acc_Plot_List) + 1]] <- plot
  
  flow_with_Exclusion_Data.Subset <- subset(flow_with_Exclusion_Data, Excluded_Identifier == identifier)
  plot <- flow_Plot(flow_with_Exclusion_Data.Subset, epoch_with_Exclusion)
  ggsave(filename = file.path(work_Dir, sprintf('Categorized_Flow.E_%s.png', epoch_with_Exclusion)), plot = plot, device = "png", width = 38, height = 12, units = "cm", dpi = 300)
  flow_with_Exclusion_Plot_List[[length(flow_with_Exclusion_Plot_List) + 1]] <- plot

  flow_without_Exclusion_Data.Subset <- subset(flow_without_Exclusion_Data, Excluded_Identifier == identifier)
  plot <- flow_Plot(flow_without_Exclusion_Data.Subset, epoch_without_Exclusion)
  ggsave(filename = file.path(work_Dir, sprintf('Categorized_Flow.E_%s.png', epoch_without_Exclusion)), plot = plot, device = "png", width = 38, height = 12, units = "cm", dpi = 300)
  flow_without_Exclusion_Plot_List[[length(flow_without_Exclusion_Plot_List) + 1]] <- plot
}


png(file.path(base_Dir, sprintf('Accuracy_Flow.IDX_%s.All.png', index)), width = 40, height = 22, res =300, units = "cm")
grid.arrange(arrangeGrob(grobs = acc_Plot_List, ncol=5))
dev.off()

png(file.path(base_Dir, sprintf('Categorized_Flow.IDX_%s.E_%s.All.png', index, epoch_with_Exclusion)), width = 120, height = 40, res =300, units = "cm")
grid.arrange(arrangeGrob(grobs =flow_with_Exclusion_Plot_List, ncol=5))
dev.off()

png(file.path(base_Dir, sprintf('Categorized_Flow.IDX_%s.E_%s.All.png', index, epoch_without_Exclusion)), width = 120, height = 20, res =300, units = "cm")
grid.arrange(arrangeGrob(grobs =flow_without_Exclusion_Plot_List, ncol=5))
dev.off()

ggsave(
  filename = file.path(base_Dir, sprintf('Accuracy_Flow.IDX_%s.Avg.png', index)),
  plot = acc_Plot(acc_Data, 'All'),
  device = "png", width = 10, height = 12, units = "cm", dpi = 300
  )
ggsave(
  filename = file.path(base_Dir, sprintf('Categorized_Flow.IDX_%s.E_%s.Avg.png', index, epoch_with_Exclusion)),
  plot = flow_Plot(data_summary(flow_with_Exclusion_Data, 'Cosine_Similarity', c('Pattern_Type', 'Category', 'Time_Step')), epoch_with_Exclusion),
  device = "png", width = 34, height = 12, units = "cm", dpi = 300
  )
ggsave(
  filename = file.path(base_Dir, sprintf('Categorized_Flow.IDX_%s.E_%s.Avg.png', index, epoch_without_Exclusion)),
  plot = flow_Plot(data_summary(flow_without_Exclusion_Data, 'Cosine_Similarity', c('Pattern_Type', 'Category', 'Time_Step')), epoch_without_Exclusion),
  device = "png", width = 34, height = 12, units = "cm", dpi = 300
  )
ggsave(
  filename = file.path(base_Dir, sprintf('Categorized_Flow.IDX_%s.E_%s.Avg.No_Pattern_Type.png', index, epoch_with_Exclusion)),
  plot = flow_Plot2(data_summary(flow_with_Exclusion_Data, 'Cosine_Similarity', c('Category', 'Time_Step')), epoch_with_Exclusion),
  device = "png", width = 10, height = 12, units = "cm", dpi = 300
)
ggsave(
  filename = file.path(base_Dir, sprintf('Categorized_Flow.IDX_%s.E_%s.Avg.No_Pattern_Type.png', index, epoch_without_Exclusion)),
  plot = flow_Plot2(data_summary(flow_without_Exclusion_Data, 'Cosine_Similarity', c('Category', 'Time_Step')), epoch_without_Exclusion),
  device = "png", width = 10, height = 12, units = "cm", dpi = 300
)

