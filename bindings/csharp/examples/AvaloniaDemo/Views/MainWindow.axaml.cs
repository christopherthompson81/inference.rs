using Avalonia.Controls;
using Avalonia.Interactivity;
using InferenceRs.AvaloniaDemo.Services;
using InferenceRs.AvaloniaDemo.ViewModels;

namespace InferenceRs.AvaloniaDemo.Views;

public partial class MainWindow : Window
{
    public MainWindow() => InitializeComponent();

    private MainViewModel Main => (MainViewModel)DataContext!;

    private async void OnSettings(object? sender, RoutedEventArgs e)
    {
        var window = new SettingsWindow();
        window.DataContext = Main.CreateSettings(new FileService(window));
        if (await window.ShowDialog<AppSettings?>(this) is { } saved) Main.ApplySettings(saved);
    }

    private void OnModels(object? sender, RoutedEventArgs e) =>
        new ModelManagerWindow { DataContext = Main.CreateModelManager() }.Show(this);
}
