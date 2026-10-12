using Avalonia.Controls;
using Avalonia.Interactivity;
using InferenceRs.AvaloniaDemo.ViewModels;

namespace InferenceRs.AvaloniaDemo.Views;

/// <summary>Closes with the saved settings, or null when cancelled.</summary>
public partial class SettingsWindow : Window
{
    public SettingsWindow() => InitializeComponent();

    private void OnSave(object? sender, RoutedEventArgs e) => Close(((SettingsViewModel)DataContext!).Save());

    private void OnCancel(object? sender, RoutedEventArgs e) => Close(null);
}
