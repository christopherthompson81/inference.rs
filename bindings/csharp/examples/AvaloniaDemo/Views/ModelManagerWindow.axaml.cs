using Avalonia.Controls;
using InferenceRs.AvaloniaDemo.ViewModels;

namespace InferenceRs.AvaloniaDemo.Views;

public partial class ModelManagerWindow : Window
{
    public ModelManagerWindow()
    {
        InitializeComponent();
        Opened += async (_, _) => await ((ModelManagerViewModel)DataContext!).RefreshCommand.ExecuteAsync(null);
        // a download still running stops with the window
        Closed += (_, _) => ((ModelManagerViewModel)DataContext!).Dispose();
    }
}
